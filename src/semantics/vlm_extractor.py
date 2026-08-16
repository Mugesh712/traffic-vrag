"""M5 — VLM attribute extraction: object crops -> structured attribute guesses.

Runs a captioning VLM on sampled per-track crops and turns the free-text
caption into {color, vehicle_type, make, model, direction, confidence}.

WHY VOCABULARY MATCHING, NOT JSON PROMPTING. Florence-2 (and small captioning
VLMs generally) are task-token driven image-to-text models, not
instruction-following chat models — there is no prompt that reliably makes
them emit "{"color": "white", ...}" on request. So the pipeline generates a
free-text caption, then scans it against small curated vocabularies. This
trades recall (an attribute phrased unusually is missed) for precision and
determinism: every extracted value comes from a fixed, known vocabulary, so
downstream stages (M6's canonical voting, M7's semantic gate) can trust the
value space without also needing to guard against schema drift or
hallucinated categories. "Parse failure" here means the caption yielded zero
matches — validated by the FrameAttributes pydantic model — and triggers one
retry with a more detailed caption task before the frame is marked uncertain.

Pluggable backends: Florence-2 is implemented (smallest, per the roadmap).
BLIP-2 / InternVL2 are registered but stubbed — same pattern as M0's CLI
subcommand stubs — so swapping backends later is a one-line config change.

Output: data/outputs/attributes_raw/<clip_id>.json (ClipRawAttributes)
Cache:  data/outputs/vlm_cache/<hash[:2]>/<hash>.json, keyed by crop content
        (not track/frame id), so identical crops are never re-captioned.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from src.semantics.vocabulary import (
    COLOR_VOCAB,
    DIRECTION_VOCAB,
    MAKE_VOCAB,
    VEHICLE_TYPE_VOCAB,
    match_vocab,
)
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.manifest import load_clip_frame_index
from src.utils.schemas import ClipAssociatedTracks, ClipRawAttributes, FrameAttributes

logger = get_logger(__name__)


class VLMExtractorError(RuntimeError):
    pass


# Relative weight of each field in the frame-level confidence score. `model`
# is excluded: Florence-2-base essentially never names a specific model, so
# including it would flatten confidence toward zero for every frame.
_CONFIDENCE_WEIGHTS = {"color": 0.4, "vehicle_type": 0.3, "direction": 0.2, "make": 0.1}


def _caption_to_fields(caption: str) -> tuple[dict[str, str | None], int]:
    fields = {
        "color": match_vocab(caption, COLOR_VOCAB),
        "vehicle_type": match_vocab(caption, VEHICLE_TYPE_VOCAB),
        "make": match_vocab(caption, MAKE_VOCAB),
        "model": None,  # needs OCR/logo reading or a stronger VLM; see M8 note below
        "direction": match_vocab(caption, DIRECTION_VOCAB),
    }
    n_found = sum(1 for k in _CONFIDENCE_WEIGHTS if fields[k] is not None)
    return fields, n_found


def _confidence(fields: dict[str, str | None]) -> float:
    return sum(weight for key, weight in _CONFIDENCE_WEIGHTS.items() if fields.get(key) is not None)


# ---------------------------------------------------------------------------
# Pluggable backends
# ---------------------------------------------------------------------------


class VLMBackend(Protocol):
    def caption(self, crops_bgr: list[np.ndarray], task: str) -> list[str]: ...


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    import torch

    # MPS deliberately excluded — see VLMConfig.device docstring.
    return "cuda" if torch.cuda.is_available() else "cpu"


class Florence2Backend:
    """Wraps microsoft/Florence-2-* via transformers' remote code.

    Not a chat model: `task` must be one of Florence-2's fixed task tokens
    (e.g. "<DETAILED_CAPTION>"), not a free-form instruction.
    """

    def __init__(self, model_id: str, device: str, max_new_tokens: int, num_beams: int):
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._Image = Image
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        ).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._no_grad = torch.no_grad

    def caption(self, crops_bgr: list[np.ndarray], task: str) -> list[str]:
        images = [self._Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = self.processor(text=[task] * len(images), images=images, return_tensors="pt").to(self.device)
        with self._no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        captions = []
        for text, image in zip(texts, images):
            parsed = self.processor.post_process_generation(text, task=task, image_size=image.size)
            captions.append(parsed.get(task, ""))
        return captions


class _UnimplementedBackend:
    def __init__(self, name: str):
        self.name = name

    def caption(self, crops_bgr: list[np.ndarray], task: str) -> list[str]:
        raise NotImplementedError(
            f"VLM backend '{self.name}' is not implemented yet; use 'florence2' "
            "(configs/pipeline.yaml: vlm.backend)."
        )


# (standard task, high-detail task) per backend. Declared outside the backend
# classes so callers can build cache keys without instantiating a model —
# otherwise a fully cached run would still pay to load the VLM.
BACKEND_TASKS: dict[str, tuple[str, str]] = {
    "florence2": ("<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"),
    "blip2": ("<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"),
    "internvl2": ("<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"),
}


def get_tasks(settings: PipelineSettings) -> tuple[str, str]:
    """(standard, high_detail) task tokens for the configured backend."""
    try:
        return BACKEND_TASKS[settings.vlm.backend]
    except KeyError:
        raise VLMExtractorError(f"Unknown VLM backend: {settings.vlm.backend}") from None


_BACKEND_CACHE: dict[str, Any] = {}


def _load_backend(settings: PipelineSettings) -> VLMBackend:
    name = settings.vlm.backend
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]

    if name == "florence2":
        device = _resolve_device(settings.vlm.device)
        backend: VLMBackend = Florence2Backend(
            model_id=settings.vlm.model_id,
            device=device,
            max_new_tokens=settings.vlm.max_new_tokens,
            num_beams=settings.vlm.num_beams,
        )
        logger.info("vlm_extractor: loaded backend=florence2 device=%s", device)
    elif name in ("blip2", "internvl2"):
        backend = _UnimplementedBackend(name)
    else:
        raise VLMExtractorError(f"Unknown VLM backend: {name}")

    _BACKEND_CACHE[name] = backend
    return backend


# ---------------------------------------------------------------------------
# Content-addressed cache
# ---------------------------------------------------------------------------


def _crop_hash(crop: np.ndarray, task: str) -> str:
    """Key on pixels AND the task token.

    Without the task in the key, a crop captioned by M5 under the standard
    task would be served back to M8 when it asks for the high-detail one --
    silently defeating the entire point of the best-shot pass.
    """
    h = hashlib.sha256()
    h.update(task.encode())
    h.update(str(crop.shape).encode())
    h.update(crop.tobytes())
    return h.hexdigest()


def _cache_path(cache_dir, crop_hash: str):
    return cache_dir / crop_hash[:2] / f"{crop_hash}.json"


def _load_cached_fields(cache_dir, crop_hash: str) -> tuple[dict[str, str | None], bool] | None:
    path = _cache_path(cache_dir, crop_hash)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload["fields"], payload.get("from_retry", False)


def _save_cached_fields(
    cache_dir,
    crop_hash: str,
    fields: dict[str, str | None],
    caption: str,
    from_retry: bool = False,
) -> None:
    path = _cache_path(cache_dir, crop_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"fields": fields, "caption": caption, "from_retry": from_retry}, indent=2)
    )


def caption_images(
    images: list[np.ndarray], settings: PipelineSettings, *, task: str
) -> list[str]:
    """Raw caption text for each image, cached by (pixels, task).

    M16's caption-RAG baseline needs the caption *prose*, not the attribute
    fields caption_crops_to_fields() parses out of it -- the whole point of
    that baseline is that it has nothing but unstructured text to retrieve
    over. Shares the same on-disk cache, so a frame captioned for one purpose
    is never re-captioned for the other.
    """
    cache_dir = settings.resolve_path(settings.vlm.cache_dir)
    hashes = [_crop_hash(image, task) for image in images]

    captions: dict[int, str] = {}
    uncached: list[int] = []
    for i, crop_hash in enumerate(hashes):
        path = _cache_path(cache_dir, crop_hash)
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("caption"):
                captions[i] = payload["caption"]
                continue
        uncached.append(i)

    if uncached:
        backend = _load_backend(settings)
        batch_size = settings.vlm.batch_size
        for start in range(0, len(uncached), batch_size):
            batch = uncached[start : start + batch_size]
            texts = backend.caption([images[i] for i in batch], task)
            for idx, text in zip(batch, texts):
                captions[idx] = text
                fields, _ = _caption_to_fields(text)
                _save_cached_fields(cache_dir, hashes[idx], fields, text, False)

    return [captions[i] for i in range(len(images))]


def caption_crops_to_fields(
    crops: list[np.ndarray],
    settings: PipelineSettings,
    *,
    task: str,
    retry_task: str | None = None,
) -> list[tuple[dict[str, str | None], bool]]:
    """Caption crops and parse them to attribute fields, using the cache.

    Returns (fields, from_retry) per crop, in input order. The backend is only
    loaded if at least one crop misses the cache. When `retry_task` is given,
    crops whose caption yielded nothing are retried once with it; M8 passes
    None, since it already asks for the most detailed caption available.

    Shared with M8 so the two VLM passes cannot drift apart in caching,
    batching or parsing behaviour.
    """
    cache_dir = settings.resolve_path(settings.vlm.cache_dir)
    hashes = [_crop_hash(crop, task) for crop in crops]

    results: dict[int, tuple[dict[str, str | None], bool]] = {}
    uncached: list[int] = []
    for i, crop_hash in enumerate(hashes):
        cached = _load_cached_fields(cache_dir, crop_hash)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append(i)

    if uncached:
        backend = _load_backend(settings)
        batch_size = settings.vlm.batch_size

        for start in range(0, len(uncached), batch_size):
            batch_idx = uncached[start : start + batch_size]
            batch_crops = [crops[i] for i in batch_idx]

            captions = backend.caption(batch_crops, task)
            fields_and_counts = [_caption_to_fields(c) for c in captions]
            from_retry = [False] * len(batch_idx)

            if retry_task is not None:
                retry_local = [i for i, (_, n) in enumerate(fields_and_counts) if n == 0]
                if retry_local:
                    retry_captions = backend.caption(
                        [batch_crops[i] for i in retry_local], retry_task
                    )
                    for local_i, retry_caption in zip(retry_local, retry_captions):
                        retry_fields, retry_n = _caption_to_fields(retry_caption)
                        if retry_n > 0:
                            fields_and_counts[local_i] = (retry_fields, retry_n)
                            captions[local_i] = retry_caption
                            from_retry[local_i] = True

            for local_i, job_idx in enumerate(batch_idx):
                fields, _ = fields_and_counts[local_i]
                results[job_idx] = (fields, from_retry[local_i])
                _save_cached_fields(
                    cache_dir, hashes[job_idx], fields, captions[local_i], from_retry[local_i]
                )

    return [results[i] for i in range(len(crops))]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


@dataclass
class _CropJob:
    track_id: str
    frame_id: str
    crop: np.ndarray
    crop_quality: float
    occlusion: float


def _crop_quality(crop: np.ndarray, settings: PipelineSettings) -> float:
    """Size x sharpness, each capped at 1.0.

    Absolute scale is arbitrary — M6 only compares weights within one track's
    votes — so this just has to be monotone in "how readable is this crop".
    """
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    size_score = min(1.0, float(np.sqrt(h * w)) / settings.vlm.quality_reference_size_px)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / settings.vlm.quality_reference_sharpness)
    return size_score * sharpness_score


def _overlap_fraction(
    bbox: tuple[float, float, float, float], others: list[tuple[float, float, float, float]]
) -> float:
    """Largest fraction of `bbox` covered by any other track's box in the frame.

    An occlusion *proxy*, and deliberately an upper bound: without depth we
    cannot tell whether the overlapping object is in front or behind. Uses
    intersection-over-own-area rather than IoU, because a large bus overlapping
    a small car scores low IoU while hiding most of the car.
    """
    x1, y1, x2, y2 = bbox
    own_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if own_area <= 0:
        return 0.0

    worst = 0.0
    for ox1, oy1, ox2, oy2 in others:
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        worst = max(worst, intersection / own_area)
    return min(1.0, worst)


def _select_sample_indices(n_available: int, n_samples: int) -> list[int]:
    """Up to n_samples indices, evenly spread across [0, n_available), so a
    track's whole lifetime is represented rather than just its first frames."""
    if n_available <= n_samples:
        return list(range(n_available))
    positions = np.linspace(0, n_available - 1, n_samples)
    return sorted(set(int(round(p)) for p in positions))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def extract_clip_attributes(clip_id: str, settings: PipelineSettings | None = None) -> ClipRawAttributes:
    settings = settings or get_settings()

    tracks_path = settings.resolve_path(settings.paths.outputs_dir) / "tracks_associated" / f"{clip_id}.json"
    if not tracks_path.exists():
        raise VLMExtractorError(
            f"No associated tracks found for {clip_id} at {tracks_path}; run `associate` first."
        )
    clip_tracks = ClipAssociatedTracks.model_validate_json(tracks_path.read_text())

    frame_paths, _, _ = load_clip_frame_index(clip_id, settings)

    # Every track's box in every frame, so a crop's occlusion can be measured
    # against its neighbours rather than guessed.
    boxes_by_frame: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for track in clip_tracks.tracks:
        for frame_id, bbox in zip(track.frames, track.bboxes):
            boxes_by_frame.setdefault(frame_id, []).append((track.track_id, tuple(bbox)))

    # One job per sampled crop, flattened across tracks so the VLM backend can
    # batch over the whole clip rather than per track.
    jobs: list[_CropJob] = []
    frame_image_cache: dict[str, np.ndarray] = {}

    for track in clip_tracks.tracks:
        indices = _select_sample_indices(len(track.frames), settings.vlm.frames_per_track)
        for i in indices:
            frame_id = track.frames[i]
            if frame_id not in frame_image_cache:
                image = cv2.imread(str(settings.resolve_path(frame_paths[frame_id])))
                if image is None:
                    logger.warning("vlm_extractor: could not read frame %s, skipping", frame_id)
                    continue
                frame_image_cache[frame_id] = image
            image = frame_image_cache[frame_id]

            raw_bbox = tuple(track.bboxes[i])
            h, w = image.shape[:2]
            x1, y1 = max(0, int(raw_bbox[0])), max(0, int(raw_bbox[1]))
            x2, y2 = min(w, int(raw_bbox[2])), min(h, int(raw_bbox[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]

            others = [
                bbox
                for other_id, bbox in boxes_by_frame.get(frame_id, [])
                if other_id != track.track_id
            ]
            jobs.append(
                _CropJob(
                    track_id=track.track_id,
                    frame_id=frame_id,
                    crop=crop,
                    crop_quality=_crop_quality(crop, settings),
                    occlusion=_overlap_fraction(raw_bbox, others),
                )
            )

    logger.info(
        "extract_clip_attributes: clip_id=%s n_tracks=%d n_sampled_crops=%d",
        clip_id,
        len(clip_tracks.tracks),
        len(jobs),
    )

    # Caption-derived fields and their provenance are cached; the geometric
    # measurements are not, since they depend on boxes rather than pixels.
    standard_task, high_detail_task = get_tasks(settings)
    results = caption_crops_to_fields(
        [job.crop for job in jobs], settings, task=standard_task, retry_task=high_detail_task
    )

    attributes = []
    for idx, job in enumerate(jobs):
        fields, was_retry = results[idx]
        attributes.append(
            FrameAttributes(
                track_id=job.track_id,
                frame_id=job.frame_id,
                **fields,
                confidence=_confidence(fields),
                crop_quality=job.crop_quality,
                occlusion=job.occlusion,
                from_retry=was_retry,
            )
        )

    clip_attributes = ClipRawAttributes(clip_id=clip_id, attributes=attributes)

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "attributes_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}.json"
    output_path.write_text(clip_attributes.model_dump_json(indent=2))

    logger.info(
        "extract_clip_attributes: clip_id=%s wrote %d frame-attributes -> %s",
        clip_id,
        len(attributes),
        output_path,
    )

    return clip_attributes
