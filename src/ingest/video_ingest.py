"""M1 — Video ingestion: MP4 -> fixed-length clips -> sampled frames with
accurate wall-clock timestamps.

Single sequential pass over the source video:
  - every frame is written into the current clip's VideoWriter
  - every Nth frame (by actual fps, not an assumed 30fps) is also saved as a
    jpg and recorded in the manifest

Output:
  data/clips/<clip_id>.mp4
  data/frames/<clip_id>/<frame_id>.jpg
  data/outputs/ingest/<video_id>_manifest.json  (VideoManifest)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import ClipManifest, FrameRecord, VideoManifest

logger = get_logger(__name__)


class VideoIngestError(RuntimeError):
    pass


def _fourcc_for(path: Path) -> int:
    return cv2.VideoWriter_fourcc(*"mp4v")


def ingest_video(
    video_path: str | Path,
    settings: PipelineSettings | None = None,
    clip_length_sec: float | None = None,
    frame_sample_interval_sec: float | None = None,
    start_wallclock: datetime | None = None,
) -> VideoManifest:
    """Split `video_path` into clips + sampled frames and write a VideoManifest.

    Timestamps are derived from `frame_index / actual_fps` (read from the
    video's own metadata via OpenCV), not an assumed frame rate, so results
    stay correct on variable/non-30fps sources.
    """
    settings = settings or get_settings()
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoIngestError(f"Video not found: {video_path}")

    clip_length_sec = clip_length_sec or settings.ingest.clip_length_sec
    frame_sample_interval_sec = (
        frame_sample_interval_sec or settings.ingest.frame_sample_interval_sec
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoIngestError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0:
        cap.release()
        raise VideoIngestError(
            f"Could not read a valid fps from {video_path} (got {fps}); "
            "refusing to guess a frame rate."
        )

    video_id = video_path.stem
    start_wallclock = start_wallclock or datetime.now()

    clip_length_frames = max(1, round(clip_length_sec * fps))
    frame_interval_frames = max(1, round(frame_sample_interval_sec * fps))

    clips_dir = settings.resolve_path(settings.paths.clips_dir)
    frames_dir = settings.resolve_path(settings.paths.frames_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ingest_video: video_id=%s fps=%.3f frame_count=%d clip_length_frames=%d "
        "frame_interval_frames=%d",
        video_id,
        fps,
        frame_count,
        clip_length_frames,
        frame_interval_frames,
    )

    clips: list[ClipManifest] = []
    writer: cv2.VideoWriter | None = None
    current_clip_idx: int | None = None
    current_clip_frames: list[FrameRecord] = []
    current_clip_first_ts = 0.0
    global_frame_idx = 0

    def _timestamp_for(idx: int) -> tuple[float, str]:
        video_ts = idx / fps
        wallclock = start_wallclock + timedelta(seconds=video_ts)
        return video_ts, wallclock.isoformat()

    def _close_current_clip(last_ts: float) -> None:
        nonlocal writer
        if writer is None or current_clip_idx is None:
            return
        writer.release()
        clip_id = f"clip_{current_clip_idx:03d}"
        _, start_iso = _timestamp_for(current_clip_idx * clip_length_frames)
        end_wallclock = start_wallclock + timedelta(seconds=last_ts)
        clips.append(
            ClipManifest(
                video_id=video_id,
                clip_id=clip_id,
                clip_path=str((clips_dir / f"{clip_id}.mp4").relative_to(
                    settings.resolve_path(".")
                )),
                start_wallclock=start_iso,
                end_wallclock=end_wallclock.isoformat(),
                fps=fps,
                resolution=(width, height),
                frames=list(current_clip_frames),
            )
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        clip_idx = global_frame_idx // clip_length_frames
        if clip_idx != current_clip_idx:
            if current_clip_idx is not None:
                prev_last_ts, _ = _timestamp_for(global_frame_idx - 1)
                _close_current_clip(prev_last_ts)
            current_clip_idx = clip_idx
            current_clip_frames = []
            clip_id = f"clip_{clip_idx:03d}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            writer = cv2.VideoWriter(str(clip_path), _fourcc_for(clip_path), fps, (width, height))
            (frames_dir / clip_id).mkdir(parents=True, exist_ok=True)

        assert writer is not None
        writer.write(frame)

        if global_frame_idx % frame_interval_frames == 0:
            clip_id = f"clip_{clip_idx:03d}"
            frame_id = f"frame_{global_frame_idx:06d}"
            frame_path = frames_dir / clip_id / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame)
            video_ts, wallclock_iso = _timestamp_for(global_frame_idx)
            current_clip_frames.append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_id=frame_id,
                    frame_path=str(frame_path.relative_to(settings.resolve_path("."))),
                    video_timestamp_sec=video_ts,
                    wallclock_time=wallclock_iso,
                    fps=fps,
                    resolution=(width, height),
                )
            )

        global_frame_idx += 1

    if current_clip_idx is not None:
        last_ts, _ = _timestamp_for(max(global_frame_idx - 1, 0))
        _close_current_clip(last_ts)

    cap.release()

    if not clips:
        raise VideoIngestError(f"No frames read from {video_path}")

    manifest = VideoManifest(
        video_id=video_id,
        source_path=str(video_path),
        fps=fps,
        resolution=(width, height),
        frame_count=global_frame_idx,
        start_wallclock=clips[0].start_wallclock,
        end_wallclock=clips[-1].end_wallclock,
        clips=clips,
    )

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "ingest"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{video_id}_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    logger.info(
        "ingest_video: wrote %d clips, %d sampled frames -> %s",
        len(clips),
        sum(len(c.frames) for c in clips),
        manifest_path,
    )

    return manifest
