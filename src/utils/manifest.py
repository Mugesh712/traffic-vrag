"""Lookup helpers over M1's ingest manifests.

Stages after M1 need a clip's real per-frame timestamps (for velocity, temporal
gates, event timing). Those live in the VideoManifest, so resolving them is
shared rather than duplicated per stage.
"""
from __future__ import annotations

from src.utils.config import PipelineSettings
from src.utils.schemas import VideoManifest


class ManifestLookupError(RuntimeError):
    pass


def load_clip_frame_index(
    clip_id: str, settings: PipelineSettings
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    """Find the VideoManifest containing `clip_id`.

    Returns (frame_id -> frame_path, frame_id -> video_timestamp_sec,
    frame_id -> wallclock_time).
    """
    ingest_dir = settings.resolve_path(settings.paths.outputs_dir) / "ingest"
    for manifest_path in sorted(ingest_dir.glob("*_manifest.json")):
        manifest = VideoManifest.model_validate_json(manifest_path.read_text())
        for clip in manifest.clips:
            if clip.clip_id == clip_id:
                paths = {f.frame_id: f.frame_path for f in clip.frames}
                timestamps = {f.frame_id: f.video_timestamp_sec for f in clip.frames}
                wallclocks = {f.frame_id: f.wallclock_time for f in clip.frames}
                return paths, timestamps, wallclocks
    raise ManifestLookupError(
        f"No ingest manifest under {ingest_dir} contains clip_id={clip_id}; run `ingest` first."
    )
