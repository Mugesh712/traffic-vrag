"""M2 — Object detection: sampled frames -> per-frame traffic-object boxes.

Wraps Ultralytics YOLOv11. Reads frames from data/frames/<video_id>/<clip_id>/,
writes data/outputs/detections/<clip_id>.json
([{frame_id, class, bbox, confidence}]).
"""
from __future__ import annotations

from pathlib import Path

import cv2

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import ClipDetections, Detection, TrafficClass

logger = get_logger(__name__)

TRAFFIC_CLASSES: set[str] = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

_MODEL_CACHE: dict[str, "YOLO"] = {}  # noqa: F821


class DetectorError(RuntimeError):
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


def _load_model(model_path: str):
    from ultralytics import YOLO

    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def _target_class_indices(model, wanted: set[str]) -> dict[int, str]:
    """Map YOLO's internal class ids -> our traffic class names, restricted
    to names we actually want (so an arbitrary model's label set doesn't
    leak non-traffic classes through)."""
    return {idx: name for idx, name in model.names.items() if name in wanted}


def detect_clip(
    clip_id: str,
    video_id: str,
    settings: PipelineSettings | None = None,
    batch_size: int = 16,
    visualize: bool = False,
) -> ClipDetections:
    settings = settings or get_settings()
    # video_id is required, not optional: this function globs the directory, so
    # a shared one silently detects on leftover frames from other videos rather
    # than failing. See the layout note in src/ingest/video_ingest.py.
    frames_dir = settings.resolve_path(settings.paths.frames_dir) / video_id / clip_id
    if not frames_dir.exists():
        raise DetectorError(f"No frames directory for clip: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise DetectorError(f"No frames found in {frames_dir}")

    model = _load_model(settings.detect.model_path)
    device = _resolve_device(settings.detect.device)
    class_map = _target_class_indices(model, TRAFFIC_CLASSES & set(TrafficClass.__args__))

    if not class_map:
        raise DetectorError(
            f"Model {settings.detect.model_path} has no classes overlapping "
            f"{TRAFFIC_CLASSES}; got names={model.names}"
        )

    logger.info(
        "detect_clip: clip_id=%s n_frames=%d device=%s conf=%.2f iou=%.2f",
        clip_id,
        len(frame_paths),
        device,
        settings.detect.conf_threshold,
        settings.detect.iou_threshold,
    )

    viz_dir = None
    if visualize:
        viz_dir = settings.resolve_path(settings.paths.outputs_dir) / "detections_viz" / clip_id
        viz_dir.mkdir(parents=True, exist_ok=True)

    detections: list[Detection] = []
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            conf=settings.detect.conf_threshold,
            iou=settings.detect.iou_threshold,
            classes=list(class_map.keys()),
            device=device,
            verbose=False,
        )

        for frame_path, result in zip(batch, results):
            frame_id = frame_path.stem
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = class_map.get(cls_id)
                if cls_name is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        **{"class": cls_name},
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                    )
                )

            if viz_dir is not None:
                annotated = result.plot()
                cv2.imwrite(str(viz_dir / frame_path.name), annotated)

    clip_detections = ClipDetections(clip_id=clip_id, detections=detections)

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}.json"
    output_path.write_text(clip_detections.model_dump_json(indent=2, by_alias=True))

    logger.info(
        "detect_clip: clip_id=%s wrote %d detections -> %s",
        clip_id,
        len(detections),
        output_path,
    )

    return clip_detections
