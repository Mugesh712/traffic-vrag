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


class LinkingConfig(BaseModel):
    appearance_similarity_threshold: float = 0.75
    motion_time_gap_max_sec: float = 120.0


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
