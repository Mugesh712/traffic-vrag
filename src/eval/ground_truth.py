"""M16 — Ground-truth schema, loader, and annotation-template generator.

WHY THIS FILE EXISTS SEPARATELY. IDF1, MOTA, ID-switch counts and attribute
accuracy are *defined* as comparisons against human-annotated truth. Without
annotations they are not merely hard to compute -- they are undefined. So the
harness treats ground truth as an explicit, typed input that is either present
(metrics computed) or absent (metrics reported as unavailable), and never
silently substitutes the system's own output for truth. Evaluating a system
against its own predictions would score every contribution at 100% and mean
nothing.

FORMAT. Per-video JSON, one file at data/ground_truth/<video_id>.json:

    {
      "video_id": "car_detection",
      "tracks": [
        {"gt_id": "gt_1", "class": "car",
         "boxes": {"frame_000062": [x1, y1, x2, y2], ...}}
      ],
      "attributes": [
        {"gt_id": "gt_1", "color": "white", "vehicle_type": "sedan",
         "make": null, "model": null}
      ]
    }

Frame ids match M1's manifest exactly, so annotations bind to the same sampled
frames the pipeline actually saw. Attribute values must come from the closed
vocabulary in src/semantics/vocabulary.py -- an annotation outside it could
never be matched by a system that can only ever emit canonical values.

ANNOTATION IS CORRECTION, NOT TRANSCRIPTION. `write_annotation_template()`
emits a template pre-filled with the pipeline's own detections so a human
corrects proposals rather than drawing every box from scratch. That is standard
practice and much faster -- but it is only legitimate because a human then
actually reviews it. A template that is accepted unedited is not ground truth,
and `load_ground_truth` refuses a file still carrying the unreviewed marker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import ClipDetections, VideoManifest

logger = get_logger(__name__)

# Written into a generated template and required to be removed by the
# annotator. Its presence means "nobody has reviewed this yet".
UNREVIEWED_MARKER = "REMOVE_THIS_KEY_ONCE_REVIEWED"


class GroundTruthError(RuntimeError):
    pass


class GTTrack(BaseModel):
    gt_id: str
    cls: str = Field(alias="class")
    # frame_id -> [x1, y1, x2, y2], only for frames where the object is visible.
    boxes: dict[str, list[float]] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class GTAttributes(BaseModel):
    gt_id: str
    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None


class GroundTruth(BaseModel):
    video_id: str
    tracks: list[GTTrack] = Field(default_factory=list)
    attributes: list[GTAttributes] = Field(default_factory=list)

    def boxes_for_frame(self, frame_id: str) -> dict[str, list[float]]:
        """gt_id -> box, for every object visible in this frame."""
        return {t.gt_id: t.boxes[frame_id] for t in self.tracks if frame_id in t.boxes}

    def attributes_by_id(self) -> dict[str, GTAttributes]:
        return {a.gt_id: a for a in self.attributes}


def ground_truth_path(video_id: str, settings: PipelineSettings) -> Path:
    return settings.resolve_path("data/ground_truth") / f"{video_id}.json"


def load_ground_truth(video_id: str, settings: PipelineSettings | None = None) -> GroundTruth | None:
    """Load annotations, or None when none exist.

    Returning None (rather than raising) is deliberate: the harness must run
    and report its GT-free metrics on an unannotated video, marking the
    GT-dependent tables unavailable rather than failing outright.
    """
    settings = settings or get_settings()
    path = ground_truth_path(video_id, settings)
    if not path.exists():
        return None

    raw = json.loads(path.read_text())
    if UNREVIEWED_MARKER in raw:
        raise GroundTruthError(
            f"{path} is still an unreviewed template: it contains "
            f"'{UNREVIEWED_MARKER}'. Correct the boxes and attributes by hand, "
            "then delete that key. Auto-generated proposals scored against the "
            "system that produced them are not ground truth."
        )
    return GroundTruth.model_validate(raw)


def write_annotation_template(
    video_id: str, settings: PipelineSettings | None = None, overwrite: bool = False
) -> Path:
    """Emit a GT template pre-filled with the pipeline's own detections.

    Each detection becomes a single-frame candidate track. The annotator's job
    is to merge boxes belonging to one real vehicle under one gt_id, fix or
    delete wrong boxes, add missed ones, and fill in attributes.
    """
    settings = settings or get_settings()
    path = ground_truth_path(video_id, settings)
    if path.exists() and not overwrite:
        raise GroundTruthError(f"{path} already exists; pass overwrite=True to replace it.")

    outputs = settings.resolve_path(settings.paths.outputs_dir)
    manifest_path = outputs / "ingest" / f"{video_id}_manifest.json"
    if not manifest_path.exists():
        raise GroundTruthError(f"No ingest manifest at {manifest_path}; run `ingest` first.")
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())

    tracks: list[dict] = []
    n = 0
    for clip in manifest.clips:
        detections_path = outputs / "detections" / f"{clip.clip_id}.json"
        if not detections_path.exists():
            continue
        clip_detections = ClipDetections.model_validate_json(detections_path.read_text())
        for det in clip_detections.detections:
            n += 1
            tracks.append(
                {
                    "gt_id": f"gt_{n}",
                    "class": det.cls,
                    "boxes": {det.frame_id: [round(v, 1) for v in det.bbox]},
                }
            )

    template = {
        UNREVIEWED_MARKER: (
            "This file was auto-generated from the pipeline's own detections and is "
            "NOT ground truth yet. Merge boxes of the same vehicle under one gt_id, "
            "fix/delete wrong boxes, add missed ones, fill in attributes, then delete "
            "this key."
        ),
        "video_id": video_id,
        "tracks": tracks,
        "attributes": [
            {"gt_id": t["gt_id"], "color": None, "vehicle_type": None, "make": None, "model": None}
            for t in tracks
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2))
    logger.info(
        "write_annotation_template: video_id=%s wrote %d candidate track(s) -> %s",
        video_id, len(tracks), path,
    )
    return path
