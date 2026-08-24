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
    clip_id: str, video_id: str, settings: PipelineSettings
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    """Frame index for `clip_id` within `video_id`'s own ingest manifest.

    video_id is required, not optional -- clip_id alone is not a unique key
    across videos (every video's first clip is "clip_000"), so this used to
    glob every manifest ever written to outputs/ingest/ and return whichever
    one matched clip_id first. On a machine that had ingested more than one
    video, that silently resolved track/associate/attribute/events frames,
    timestamps, and image paths against an UNRELATED video's manifest.

    Observed live: a real run's tracker read frame data from a stale earlier
    job's manifest (different resolution, different frame count, pre-M2-fix
    flat paths). Consecutive detections of the same car were then never
    actually consecutive in time as far as the frame index was concerned, so
    IOU-based frame-to-frame matching failed across the board and every
    object came out as an orphaned single-frame track -- looking exactly like
    "the model only recognizes one car per frame", even though detection
    itself (M2) was finding 7-13 cars per frame correctly the whole time.
    """
    manifest_path = settings.resolve_path(settings.paths.outputs_dir) / "ingest" / f"{video_id}_manifest.json"
    if not manifest_path.exists():
        raise ManifestLookupError(
            f"No ingest manifest for video_id={video_id} at {manifest_path}; run `ingest` first."
        )
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())
    for clip in manifest.clips:
        if clip.clip_id == clip_id:
            paths = {f.frame_id: f.frame_path for f in clip.frames}
            timestamps = {f.frame_id: f.video_timestamp_sec for f in clip.frames}
            wallclocks = {f.frame_id: f.wallclock_time for f in clip.frames}
            return paths, timestamps, wallclocks
    raise ManifestLookupError(f"video_id={video_id}'s manifest has no clip_id={clip_id}.")
