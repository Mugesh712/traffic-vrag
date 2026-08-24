"""Regression tests for clip_id collisions between videos.

Clip ids restart at clip_000 for every video, so anything keyed on clip_id
alone can silently resolve against the WRONG video once more than one has
ever been processed on the same machine.

The first class of this bug (frames written to a shared data/frames/<clip_id>/
directory, and M2 globbing that directory) is covered below. A second,
downstream instance of the same root cause was found later: load_clip_frame_
index (used by M3/M4/M5/M9) globbed every ingest manifest on disk and returned
whichever one matched clip_id first, rather than the current video's own --
observed live as a real tracking run silently reading a stale, unrelated
video's frame data, breaking IOU-based frame-to-frame matching so badly that
every object came out as an orphaned single-frame track. Neither function had
any coverage, which is why both bugs survived. These tests pin the layout and
both functions' use of it.

Fixtures are written under the real data/ tree (resolve_path is anchored to the
project root, the same constraint the M3/M4/M5 tests work within) and cleaned up
unconditionally afterward.
"""
from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from src.ingest.video_ingest import ingest_video
from src.perception.detector import DetectorError, detect_clip
from src.utils.config import get_settings
from src.utils.manifest import ManifestLookupError, load_clip_frame_index

VIDEO_A = "video_test_ns_a"
VIDEO_B = "video_test_ns_b"


def _write_video(path, n_frames: int, fill: int) -> None:
    """A tiny synthetic clip. Content is irrelevant here -- only the frame
    count matters, since these tests assert on which files land where."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32)
    )
    frame = np.full((32, 32, 3), fill, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture
def two_videos():
    settings = get_settings()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    frames_dir = settings.resolve_path(settings.paths.frames_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately different lengths: an equal-length pair would hide the bug,
    # because every stale frame would be overwritten by the second ingest.
    paths = {VIDEO_A: raw_dir / f"{VIDEO_A}.mp4", VIDEO_B: raw_dir / f"{VIDEO_B}.mp4"}
    _write_video(paths[VIDEO_A], n_frames=40, fill=10)
    _write_video(paths[VIDEO_B], n_frames=10, fill=200)

    try:
        yield settings, paths, frames_dir
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)
        for vid in (VIDEO_A, VIDEO_B):
            shutil.rmtree(frames_dir / vid, ignore_errors=True)
            manifest = (
                settings.resolve_path(settings.paths.outputs_dir)
                / "ingest"
                / f"{vid}_manifest.json"
            )
            manifest.unlink(missing_ok=True)
            (settings.resolve_path(settings.paths.clips_dir) / f"{vid}.mp4").unlink(
                missing_ok=True
            )


def test_frames_are_namespaced_per_video(two_videos):
    """Two videos whose clips share an id must not share a frame directory."""
    settings, paths, frames_dir = two_videos

    manifest_a = ingest_video(paths[VIDEO_A], settings=settings)
    manifest_b = ingest_video(paths[VIDEO_B], settings=settings)

    # Both produce a clip_000, which is exactly the collision condition.
    assert manifest_a.clips[0].clip_id == "clip_000"
    assert manifest_b.clips[0].clip_id == "clip_000"

    dir_a = frames_dir / VIDEO_A / "clip_000"
    dir_b = frames_dir / VIDEO_B / "clip_000"
    assert dir_a.is_dir() and dir_b.is_dir()
    assert dir_a != dir_b

    # The shorter second ingest must not leave the longer first one's frames
    # visible to it -- the exact failure that inflated M2's frame count.
    assert len(list(dir_a.glob("*.jpg"))) == len(manifest_a.clips[0].frames)
    assert len(list(dir_b.glob("*.jpg"))) == len(manifest_b.clips[0].frames)
    assert len(list(dir_a.glob("*.jpg"))) > len(list(dir_b.glob("*.jpg")))


def test_manifest_frame_paths_carry_the_video_id(two_videos):
    """Downstream stages resolve frames from these recorded paths rather than
    globbing, so the video id has to be in them."""
    settings, paths, _ = two_videos

    manifest = ingest_video(paths[VIDEO_A], settings=settings)

    frame_paths = [f.frame_path for f in manifest.clips[0].frames]
    assert frame_paths
    assert all(VIDEO_A in p for p in frame_paths)


def test_detect_clip_looks_under_the_video_id(two_videos):
    """M2 must resolve its frames per video. Asserted via the error path so the
    test stays fast: the directory check runs before the YOLO model loads, so
    no weights are needed."""
    settings, paths, _ = two_videos

    ingest_video(paths[VIDEO_A], settings=settings)

    with pytest.raises(DetectorError) as excinfo:
        detect_clip("clip_000", "video_that_was_never_ingested", settings=settings)

    # Resolving to a shared, video-less directory would have found video A's
    # frames and silently detected on them instead of raising.
    assert "video_that_was_never_ingested" in str(excinfo.value)


def test_load_clip_frame_index_resolves_the_correct_video_when_ambiguous(two_videos):
    """Two videos both have a clip_000; asking for video A's must never
    silently hand back video B's frame index just because it happened to be
    the one whose manifest sorted first (alphabetically or otherwise)."""
    settings, paths, _ = two_videos

    manifest_a = ingest_video(paths[VIDEO_A], settings=settings)
    manifest_b = ingest_video(paths[VIDEO_B], settings=settings)

    frame_paths, _, _ = load_clip_frame_index("clip_000", VIDEO_A, settings=settings)

    assert len(frame_paths) == len(manifest_a.clips[0].frames)
    assert len(frame_paths) != len(manifest_b.clips[0].frames)  # the two fixtures differ on purpose
    assert all(VIDEO_A in p for p in frame_paths.values())
    assert not any(VIDEO_B in p for p in frame_paths.values())


def test_load_clip_frame_index_raises_for_an_unknown_video_id(two_videos):
    """Falling back to whatever manifest exists on disk would have found
    video A's frames and silently returned them for a video that was never
    ingested, instead of raising."""
    settings, paths, _ = two_videos

    ingest_video(paths[VIDEO_A], settings=settings)

    with pytest.raises(ManifestLookupError) as excinfo:
        load_clip_frame_index("clip_000", "video_that_was_never_ingested", settings=settings)

    assert "video_that_was_never_ingested" in str(excinfo.value)
