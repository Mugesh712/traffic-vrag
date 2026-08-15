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
    best_shot_scores: list[float] = Field(default_factory=list)  # aligned with best_shot_crops

    model_config = {"populate_by_name": True}


class ClipTracks(BaseModel):
    clip_id: str
    tracks: list[Track] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M4 — Intra-clip association
# ---------------------------------------------------------------------------


class MergeLogEntry(BaseModel):
    """One accepted fragment link (from_track_id -> to_track_id), with the
    score every gate produced. Per-link rather than per-component so ablations
    can attribute each repair to its evidence."""

    merged_track_ids: list[str]  # [from_track_id, to_track_id]
    into_track_id: str  # surviving id of the component both ended up in
    gate_scores: dict[str, float]


class RejectedLinkEntry(BaseModel):
    """A temporally plausible pair that did not become a merge. Only pairs
    passing the temporal gate are logged — logging every O(n^2) pair would swamp
    the file with trivially distant ones."""

    from_track_id: str
    to_track_id: str
    failed_gate: str  # first gate to fail, under the documented gate order
    gate_scores: dict[str, float]


class ClipAssociatedTracks(BaseModel):
    clip_id: str
    tracks: list[Track] = Field(default_factory=list)
    merge_log: list[MergeLogEntry] = Field(default_factory=list)
    # Failed a hard gate.
    rejected_links: list[RejectedLinkEntry] = Field(default_factory=list)
    # Passed every gate but lost a contested endpoint to a higher-scoring link
    # (failed_gate == "assignment_conflict"). Kept separate from rejected_links
    # so ablations can tell "the gates rejected it" from "the matcher did".
    suppressed_links: list[RejectedLinkEntry] = Field(default_factory=list)


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
    # Caption *coverage* (how many fields the caption yielded), not a
    # per-attribute certainty. M6 weights votes by provenance and crop
    # measurements instead; this is kept as a caption-richness signal.
    confidence: float = 0.0
    # Inputs to M6's vote weighting, measured where the crop already exists.
    crop_quality: float = 0.0  # size x sharpness, each capped at 1.0
    occlusion: float = 0.0  # fraction of this box covered by another track's box
    from_retry: bool = False  # values came from the speculative retry caption


class ClipRawAttributes(BaseModel):
    clip_id: str
    attributes: list[FrameAttributes] = Field(default_factory=list)


class VoteDistribution(BaseModel):
    attribute: str
    winner: Optional[str] = None
    # Weight shares summing to 1.0, highest first (e.g. white 0.7, silver 0.3).
    distribution: dict[str, float] = Field(default_factory=dict)
    uncertain: bool = False
    n_votes: int = 0  # observations that supplied a value for this attribute
    margin: float = 0.0  # winner's share minus runner-up's
    # "no_evidence" | "insufficient_evidence" | "low_margin" | None. Makes the
    # M16 ablation ("how often does the system admit uncertainty, and why?")
    # readable straight off the output.
    uncertain_reason: Optional[str] = None


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


class RejectedGlobalLinkEntry(BaseModel):
    """A cross-clip pair that did not become a link. Same three-log structure
    as M4: `failed_gate` names the first gate to fail, or
    "assignment_conflict" when the pair cleared every gate but the Hungarian
    solver assigned one of its endpoints elsewhere."""

    from_clip_id: str
    from_track_id: str
    to_clip_id: str
    to_track_id: str
    failed_gate: str
    gate_scores: dict[str, float] = Field(default_factory=dict)


class MasterObjectIndex(BaseModel):
    video_id: str = ""
    objects: list[GlobalObject] = Field(default_factory=list)
    # Failed a hard gate.
    rejected_links: list[RejectedGlobalLinkEntry] = Field(default_factory=list)
    # Cleared every gate but lost its endpoint in the assignment.
    suppressed_links: list[RejectedGlobalLinkEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M8 — Global best-shot confirmation
# ---------------------------------------------------------------------------


class FinalAttribute(BaseModel):
    attribute: str
    value: Optional[str] = None
    confidence: float = 0.0
    # "agreed" | "clip_voting" | "best_shot" | "unconfirmed"
    source: str = "unconfirmed"
    uncertain: bool = True


class CorrectionLogEntry(BaseModel):
    """One attribute where the best-shot pass changed or confirmed the
    clip-level answer. Counted per outcome, this is a paper result: it
    quantifies how much semantic correction the system actually performs.

    `filled` (best shot answered where clip-level voting was uncertain) is
    kept distinct from `corrected` (best shot overturned a confident answer) --
    they are different claims and should not be reported as one total.
    """

    global_id: str
    attribute: str
    clip_level_value: Optional[str] = None
    best_shot_value: Optional[str] = None
    final_value: Optional[str] = None
    # "confirmed" | "filled" | "corrected" | "retained" | "unconfirmed"
    outcome: str


class GlobalObjectFinal(BaseModel):
    global_id: str
    cls: TrafficClass = Field(alias="class")
    sightings: list[Sighting] = Field(default_factory=list)
    attributes: list[FinalAttribute] = Field(default_factory=list)
    best_shot_crops: list[str] = Field(default_factory=list)  # the crops actually re-read

    model_config = {"populate_by_name": True}


class FinalObjectIndex(BaseModel):
    video_id: str = ""
    objects: list[GlobalObjectFinal] = Field(default_factory=list)
    correction_log: list[CorrectionLogEntry] = Field(default_factory=list)


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
