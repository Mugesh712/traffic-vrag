"""M3 — Multi-object tracking: per-frame detections -> persistent tracks with
OSNet appearance embeddings.

Wraps ByteTrack (supervision) for temporal association and OSNet (torchreid,
ImageNet-pretrained) for appearance embeddings. Feeds M2's detections; frame
images and accurate per-frame timestamps come from M1's video manifest, so
velocity is computed from real elapsed time rather than an assumed frame rate.

Output: data/outputs/tracks/<clip_id>.json (ClipTracks)
Crops:  data/crops/<clip_id>/<track_id>/<frame_id>.jpg (top-K best-shot candidates only)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import supervision as sv

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import ClipDetections, ClipTracks, Track, TrafficClass, VideoManifest

logger = get_logger(__name__)

_CLASS_NAMES: list[str] = list(TrafficClass.__args__)
_CLASS_TO_ID = {name: i for i, name in enumerate(_CLASS_NAMES)}

_EXTRACTOR_CACHE: dict[str, Any] = {}


class TrackerError(RuntimeError):
    pass


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_reid_extractor(model_name: str, device: str):
    key = f"{model_name}:{device}"
    if key not in _EXTRACTOR_CACHE:
        from torchreid.reid.utils import FeatureExtractor

        _EXTRACTOR_CACHE[key] = FeatureExtractor(model_name=model_name, model_path="", device=device)
    return _EXTRACTOR_CACHE[key]


def _extract_embeddings(extractor, crops_bgr: list[np.ndarray]) -> np.ndarray:
    rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
    feats = extractor(rgb_crops)
    return feats.cpu().numpy() if hasattr(feats, "cpu") else np.asarray(feats)


def _load_video_manifest_for_clip(
    clip_id: str, settings: PipelineSettings
) -> tuple[dict[str, str], dict[str, float]]:
    """Find the VideoManifest containing `clip_id` and return
    (frame_id -> frame_path, frame_id -> video_timestamp_sec)."""
    ingest_dir = settings.resolve_path(settings.paths.outputs_dir) / "ingest"
    for manifest_path in sorted(ingest_dir.glob("*_manifest.json")):
        manifest = VideoManifest.model_validate_json(manifest_path.read_text())
        for clip in manifest.clips:
            if clip.clip_id == clip_id:
                paths = {f.frame_id: f.frame_path for f in clip.frames}
                timestamps = {f.frame_id: f.video_timestamp_sec for f in clip.frames}
                return paths, timestamps
    raise TrackerError(
        f"No ingest manifest under {ingest_dir} contains clip_id={clip_id}; run `ingest` first."
    )


def _crop(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _sharpness(crop: np.ndarray) -> float:
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quality_score(bbox: tuple[float, float, float, float], confidence: float, sharpness: float) -> float:
    """Size x confidence x (capped) sharpness. A full occlusion estimate is
    deferred to M8, which re-scores best shots across the whole global object."""
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area * confidence * min(sharpness, 500.0)


class _TrackAccumulator:
    def __init__(self, track_id: str, embedding_top_k: int, ema_alpha: float):
        self.track_id = track_id
        self.embedding_top_k = embedding_top_k
        self.ema_alpha = ema_alpha
        self.class_votes: dict[str, int] = defaultdict(int)
        self.frames: list[str] = []
        self.bboxes: list[tuple[float, float, float, float]] = []
        self.centers: list[tuple[float, float]] = []
        self.timestamps: list[float] = []
        self._smoothed_embedding: np.ndarray | None = None
        self._all_crop_paths: list[str] = []
        self._best_crops: list[tuple[float, str]] = []  # (quality, path), kept sorted desc, len <= top_k

    def add(
        self,
        frame_id: str,
        cls_name: str,
        bbox: tuple[float, float, float, float],
        timestamp_sec: float,
        embedding: np.ndarray,
        quality: float,
        crop_path: str,
    ) -> None:
        self.class_votes[cls_name] += 1
        self.frames.append(frame_id)
        self.bboxes.append(bbox)
        self.centers.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        self.timestamps.append(timestamp_sec)
        self._all_crop_paths.append(crop_path)

        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            unit = embedding / norm
            self._smoothed_embedding = (
                unit
                if self._smoothed_embedding is None
                else self.ema_alpha * self._smoothed_embedding + (1 - self.ema_alpha) * unit
            )

        self._best_crops.append((quality, crop_path))
        self._best_crops.sort(key=lambda t: t[0], reverse=True)
        del self._best_crops[self.embedding_top_k :]

    @property
    def all_crop_paths(self) -> list[str]:
        return self._all_crop_paths

    def finalize(self) -> Track:
        velocity: list[tuple[float, float]] = [(0.0, 0.0)]
        for i in range(1, len(self.centers)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            if dt <= 0:
                velocity.append((0.0, 0.0))
                continue
            dx = self.centers[i][0] - self.centers[i - 1][0]
            dy = self.centers[i][1] - self.centers[i - 1][1]
            velocity.append((dx / dt, dy / dt))

        dominant_direction_deg = None
        if len(self.centers) >= 2:
            dx = self.centers[-1][0] - self.centers[0][0]
            dy = self.centers[-1][1] - self.centers[0][1]
            if dx != 0.0 or dy != 0.0:
                dominant_direction_deg = float(np.degrees(np.arctan2(dy, dx)))

        cls_name = max(self.class_votes.items(), key=lambda kv: kv[1])[0]
        embedding = self._smoothed_embedding.tolist() if self._smoothed_embedding is not None else []

        return Track(
            track_id=self.track_id,
            **{"class": cls_name},
            frames=self.frames,
            bboxes=self.bboxes,
            centers=self.centers,
            velocity=velocity,
            dominant_direction_deg=dominant_direction_deg,
            embedding=embedding,
            best_shot_crops=[path for _, path in self._best_crops],
        )


def track_clip(clip_id: str, settings: PipelineSettings | None = None) -> ClipTracks:
    settings = settings or get_settings()

    detections_path = settings.resolve_path(settings.paths.outputs_dir) / "detections" / f"{clip_id}.json"
    if not detections_path.exists():
        raise TrackerError(f"No detections found for {clip_id} at {detections_path}; run `detect` first.")
    clip_detections = ClipDetections.model_validate_json(detections_path.read_text())

    frame_paths, frame_timestamps = _load_video_manifest_for_clip(clip_id, settings)

    detections_by_frame: dict[str, list] = defaultdict(list)
    for det in clip_detections.detections:
        detections_by_frame[det.frame_id].append(det)

    device = _resolve_device(settings.track.reid_device)
    extractor = _load_reid_extractor(settings.track.reid_model, device)

    frame_rate = 1.0 / settings.ingest.frame_sample_interval_sec
    tracker = sv.ByteTrack(
        track_activation_threshold=settings.track.track_activation_threshold,
        lost_track_buffer=settings.track.lost_track_buffer_frames,
        minimum_matching_threshold=settings.track.minimum_matching_threshold,
        frame_rate=frame_rate,
        minimum_consecutive_frames=settings.track.minimum_consecutive_frames,
    )

    crops_dir = settings.resolve_path(settings.paths.crops_dir) / clip_id
    accumulators: dict[str, _TrackAccumulator] = {}

    ordered_frame_ids = sorted(frame_paths.keys())
    logger.info(
        "track_clip: clip_id=%s n_frames=%d n_detections=%d device=%s",
        clip_id,
        len(ordered_frame_ids),
        len(clip_detections.detections),
        device,
    )

    for frame_id in ordered_frame_ids:
        dets = detections_by_frame.get(frame_id, [])
        image_path = settings.resolve_path(frame_paths[frame_id])
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("track_clip: could not read frame %s, skipping", image_path)
            continue

        if dets:
            xyxy = np.array([d.bbox for d in dets], dtype=float)
            confidence = np.array([d.confidence for d in dets], dtype=float)
            class_id = np.array([_CLASS_TO_ID[d.cls] for d in dets], dtype=int)
        else:
            xyxy = np.zeros((0, 4), dtype=float)
            confidence = np.zeros((0,), dtype=float)
            class_id = np.zeros((0,), dtype=int)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = tracker.update_with_detections(sv_detections)

        if len(tracked) == 0:
            continue

        crops = [_crop(image, tuple(box)) for box in tracked.xyxy]
        embeddings = _extract_embeddings(extractor, crops)

        timestamp = frame_timestamps[frame_id]
        for i in range(len(tracked)):
            track_id = str(int(tracked.tracker_id[i]))
            cls_name = _CLASS_NAMES[int(tracked.class_id[i])]
            bbox = tuple(float(v) for v in tracked.xyxy[i])
            crop = crops[i]
            sharpness = _sharpness(crop)
            quality = _quality_score(bbox, float(tracked.confidence[i]), sharpness)

            crop_dir = crops_dir / track_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(crop_path), crop)

            acc = accumulators.setdefault(
                track_id,
                _TrackAccumulator(track_id, settings.track.embedding_top_k, settings.track.embedding_ema_alpha),
            )
            acc.add(
                frame_id=frame_id,
                cls_name=cls_name,
                bbox=bbox,
                timestamp_sec=timestamp,
                embedding=embeddings[i],
                quality=quality,
                crop_path=str(crop_path.relative_to(settings.resolve_path("."))),
            )

    tracks = [acc.finalize() for acc in accumulators.values()]

    # Only the top-K best-shot crops per track are worth keeping on disk.
    for acc, track in zip(accumulators.values(), tracks):
        kept = set(track.best_shot_crops)
        for path in acc.all_crop_paths:
            if path not in kept:
                (settings.resolve_path(".") / path).unlink(missing_ok=True)

    clip_tracks = ClipTracks(clip_id=clip_id, tracks=tracks)

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}.json"
    output_path.write_text(clip_tracks.model_dump_json(indent=2, by_alias=True))

    logger.info(
        "track_clip: clip_id=%s wrote %d tracks -> %s",
        clip_id,
        len(tracks),
        output_path,
    )

    return clip_tracks
