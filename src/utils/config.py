"""Config-driven settings loaded from configs/pipeline.yaml.

Every module should import `get_settings()` rather than hardcoding thresholds.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    clips_dir: str = "data/clips"
    frames_dir: str = "data/frames"
    crops_dir: str = "data/crops"
    outputs_dir: str = "data/outputs"


class IngestConfig(BaseModel):
    clip_length_sec: float = 30.0
    frame_sample_interval_sec: float = 1.0


class DetectConfig(BaseModel):
    model_path: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: list[str] = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    device: str = "auto"


class TrackConfig(BaseModel):
    tracker: str = "bytetrack"
    track_activation_threshold: float = 0.25
    lost_track_buffer_frames: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1
    reid_model: str = "osnet_x0_25"
    reid_device: str = "auto"
    embedding_ema_alpha: float = 0.9
    embedding_top_k: int = 5


class AssociationScoreWeights(BaseModel):
    appearance: float = 0.5
    motion: float = 0.3
    temporal: float = 0.2


class AssociationConfig(BaseModel):
    appearance_similarity_threshold: float = 0.7
    max_gap_sec: float = 3.0
    stationary_variance_threshold: float = 5.0
    stationary_iou_threshold: float = 0.3
    motion_tolerance: float = 2.5
    velocity_window: int = 3
    score_weights: AssociationScoreWeights = AssociationScoreWeights()


class VLMConfig(BaseModel):
    backend: str = "florence2"
    model_id: str = "microsoft/Florence-2-base"
    # "auto" resolves to cuda if available, else cpu. MPS is deliberately
    # excluded: Florence-2's custom remote-code ops hang on Apple's MPS
    # backend (observed: single caption exceeded 10 minutes vs ~2s on CPU).
    device: str = "auto"
    max_new_tokens: int = 200
    num_beams: int = 3
    batch_size: int = 8
    frames_per_track: int = 8
    cache_dir: str = "data/outputs/vlm_cache"
    # Crop-quality references. A crop at or above these is scored 1.0; the
    # scale is arbitrary because M6 only ever compares weights *within* one
    # track's votes.
    quality_reference_size_px: float = 96.0
    quality_reference_sharpness: float = 200.0


class VotingConfig(BaseModel):
    confidence_margin_threshold: float = 0.15
    min_evidence_count: int = 3
    # Values recovered only by the speculative <MORE_DETAILED_CAPTION> retry
    # are weaker evidence than ones the primary caption produced.
    retry_confidence: float = 0.6


class LinkingScoreWeights(BaseModel):
    appearance: float = 0.4
    motion: float = 0.3
    semantic: float = 0.3


class LinkingConfig(BaseModel):
    appearance_similarity_threshold: float = 0.75
    # Clips are contiguous, so a genuine cross-boundary link spans about one
    # frame interval. A large budget here would let the gate bless a track
    # that vanished mid-clip -- exactly the false link it exists to stop.
    max_gap_sec: float = 5.0
    motion_tolerance: float = 3.0  # prediction error, in object diagonals
    velocity_window: int = 3
    # Only adjacent clips by default: motion extrapolation degrades fast, and
    # a vehicle reappearing several clips later is a re-entry, not a
    # continuation. Raising this trades under-linking for invented identity.
    max_clip_distance: int = 1
    # Ablation switches for M16 ("remove the semantic gate from M7").
    enable_semantic_gate: bool = True
    enable_appearance_gate: bool = True
    score_weights: LinkingScoreWeights = LinkingScoreWeights()


class ConfirmationConfig(BaseModel):
    """M8 global best-shot confirmation."""

    top_k: int = 3  # best shots re-read with the high-detail prompt
    quality_reference_size_px: float = 128.0
    quality_reference_sharpness: float = 200.0
    # Viewpoint preference, as a blend: 0.0 disables it entirely, 1.0 scores
    # purely on frontality. A proxy from aspect ratio, not a real viewpoint
    # classifier -- M16 should check whether it earns its weight.
    viewpoint_weight: float = 0.5
    frontal_aspect_ratio: float = 1.2  # w/h at or below this reads as frontal
    side_aspect_ratio: float = 2.2  # at or above this reads as side-on
    # A tiny, deliberately high-quality sample: one excellent crop is evidence.
    min_evidence_count: int = 1
    confidence_margin_threshold: float = 0.15
    confidence_boost: float = 0.1  # applied when best shot agrees
    disagreement_penalty: float = 0.2  # applied when clip-level overrules best shot


class EventsConfig(BaseModel):
    """M9 rule-based event detection. All thresholds are in pixels/sec and
    seconds, on each global object's stitched real-timestamp trajectory."""

    stop_speed_threshold_px_s: float = 5.0  # below this, considered stopped
    stop_min_duration_sec: float = 2.0
    moving_speed_threshold_px_s: float = 8.0  # above this, heading is trusted
    turn_min_degrees: float = 45.0
    turn_min_window_sec: float = 0.5  # guards against single-sample noise
    turn_max_window_sec: float = 8.0
    lane_change_min_lateral_px: float = 40.0
    lane_change_max_heading_deg: float = 20.0  # heading must stay roughly constant
    lane_change_min_window_sec: float = 0.5
    lane_change_max_window_sec: float = 6.0
    overtake_max_heading_diff_deg: float = 30.0  # "same direction" gate
    overtake_min_lateral_px: float = 15.0  # "lateral displacement present" gate
    overtake_min_sustain_sec: float = 1.0
    overtake_max_window_sec: float = 15.0
    max_evidence_frames: int = 5
    regions_dir: str = "configs/regions"  # per-video line/region YAML, optional


class Neo4jConfig(BaseModel):
    """M10 knowledge graph connection. Override the password out of band with
    PIPELINE__NEO4J__PASSWORD rather than committing it here."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "traffic-vrag"
    database: str = "neo4j"
    batch_size: int = 1000


class VectorStoreConfig(BaseModel):
    """M11 ChromaDB vector store."""

    persist_dir: str = "data/outputs/vector_store"
    object_collection: str = "object_timelines"
    event_collection: str = "events"


class RetrievalWeights(BaseModel):
    """Fusion weights. All three terms are normalized to [0,1] before
    weighting, so these are directly interpretable against each other."""

    vector: float = 0.5  # graded semantic relevance
    graph: float = 0.35  # exact constraint satisfaction, authoritative
    confidence: float = 0.15  # the system's own certainty in matched attributes


class RetrievalConfig(BaseModel):
    """M12 hybrid retrieval."""

    top_k: int = 5
    vector_candidates: int = 20
    graph_limit: int = 50
    # Reciprocal Rank Fusion constant. Rank-based fusion needs no score
    # calibration between vector distance and graph membership, which are not
    # on a common scale.
    rrf_k: int = 60
    # Counterfactual and forecast questions need surrounding context to reason
    # over, not the single best match, so their result set is widened.
    reasoning_widen_factor: int = 2
    intent_backend: str = "rules"  # "rules" | "llm" (M13 supplies the LLM one)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "data/outputs/logs"


class PipelineSettings(BaseSettings):
    """Root settings object. Values are loaded from YAML and can be
    overridden by environment variables prefixed with PIPELINE__ (e.g.
    PIPELINE__DETECT__CONF_THRESHOLD=0.5), using pydantic-settings'
    nested-delimiter convention.
    """

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE__",
        env_nested_delimiter="__",
    )

    paths: PathsConfig = PathsConfig()
    ingest: IngestConfig = IngestConfig()
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    association: AssociationConfig = AssociationConfig()
    vlm: VLMConfig = VLMConfig()
    voting: VotingConfig = VotingConfig()
    linking: LinkingConfig = LinkingConfig()
    confirmation: ConfirmationConfig = ConfirmationConfig()
    events: EventsConfig = EventsConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    logging: LoggingConfig = LoggingConfig()

    def resolve_path(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


def _load_yaml(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineSettings:
    """Load and cache pipeline settings from YAML, with env var overrides."""
    raw = _load_yaml(Path(config_path))
    return PipelineSettings(**raw)
