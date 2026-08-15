"""Pydantic schemas for every JSON artifact passed between pipeline stages.

Rule: modules never pass Python objects to each other. Each stage writes one
of these schemas to data/outputs/<stage>/<clip_id>.json, and the next stage
reads and validates it back from disk.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# M1 — Ingestion
# ---------------------------------------------------------------------------


class FrameRecord(BaseModel):
    clip_id: str
    frame_id: str
    frame_path: str
    video_timestamp_sec: float
    wallclock_time: str
    fps: float
    resolution: tuple[int, int]


class ClipManifest(BaseModel):
    video_id: str
    clip_id: str
    clip_path: str
    start_wallclock: str
    end_wallclock: str
    fps: float
    resolution: tuple[int, int]
    frames: list[FrameRecord] = Field(default_factory=list)


class VideoManifest(BaseModel):
    video_id: str
    source_path: str
    fps: float
    resolution: tuple[int, int]
    frame_count: int
    start_wallclock: str
    end_wallclock: str
    clips: list[ClipManifest] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M2 — Detection
# ---------------------------------------------------------------------------

TrafficClass = Literal["car", "truck", "bus", "motorcycle", "bicycle", "person"]


class Detection(BaseModel):
    frame_id: str
    cls: TrafficClass = Field(alias="class")
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float

    model_config = {"populate_by_name": True}


class ClipDetections(BaseModel):
    clip_id: str
    detections: list[Detection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M3 — Tracking
# ---------------------------------------------------------------------------


class Track(BaseModel):
    track_id: str
    cls: TrafficClass = Field(alias="class")
    frames: list[str] = Field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = Field(default_factory=list)
    centers: list[tuple[float, float]] = Field(default_factory=list)
    velocity: list[tuple[float, float]] = Field(default_factory=list)  # px/sec, aligned with frames
    dominant_direction_deg: Optional[float] = None  # atan2(dy, dx) of net displacement, image coords
    embedding: list[float] = Field(default_factory=list)  # rolling-average OSNet embedding, L2-normalized
    best_shot_crops: list[str] = Field(default_factory=list)  # paths, highest quality first

    model_config = {"populate_by_name": True}


class ClipTracks(BaseModel):
    clip_id: str
    tracks: list[Track] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M4 — Intra-clip association
# ---------------------------------------------------------------------------


class MergeLogEntry(BaseModel):
    merged_track_ids: list[str]
    into_track_id: str
    gate_scores: dict[str, float]


class ClipAssociatedTracks(BaseModel):
    clip_id: str
    tracks: list[Track] = Field(default_factory=list)
    merge_log: list[MergeLogEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M5/M6 — Attributes
# ---------------------------------------------------------------------------


class FrameAttributes(BaseModel):
    track_id: str
    frame_id: str
    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    direction: Optional[str] = None
    confidence: float = 0.0


class ClipRawAttributes(BaseModel):
    clip_id: str
    attributes: list[FrameAttributes] = Field(default_factory=list)


class VoteDistribution(BaseModel):
    attribute: str
    winner: Optional[str] = None
    distribution: dict[str, float] = Field(default_factory=dict)
    uncertain: bool = False


class CanonicalAttributes(BaseModel):
    track_id: str
    votes: list[VoteDistribution] = Field(default_factory=list)


class ClipCanonicalAttributes(BaseModel):
    clip_id: str
    tracks: list[CanonicalAttributes] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M7 — Global linking
# ---------------------------------------------------------------------------


class Sighting(BaseModel):
    clip_id: str
    track_id: str
    first_seen: str
    last_seen: str
    gate_scores: dict[str, float] = Field(default_factory=dict)


class GlobalObject(BaseModel):
    global_id: str
    cls: TrafficClass = Field(alias="class")
    sightings: list[Sighting] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MasterObjectIndex(BaseModel):
    objects: list[GlobalObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M9 — Events
# ---------------------------------------------------------------------------

EventType = Literal["OVERTAKE", "STOP", "PARK", "TURN", "LANE_CHANGE", "CROSSES"]


class Event(BaseModel):
    event_id: str
    type: EventType
    subject_id: str
    object_id: Optional[str] = None
    start_time: str
    end_time: str
    confidence: float
    evidence_frames: list[str] = Field(default_factory=list)


class EventLog(BaseModel):
    events: list[Event] = Field(default_factory=list)
