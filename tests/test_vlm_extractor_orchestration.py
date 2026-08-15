"""Integration-level test for M5's orchestration: retry-on-empty-parse and the
content-addressed cache, using a fake backend so no real VLM is loaded.

Writes fixtures under the real data/ tree (resolve_path is anchored to the
actual project root, same constraint M3/M4's tests work within) and cleans
up unconditionally afterward.
"""
from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest

import src.semantics.vlm_extractor as vlm_extractor
from src.semantics.vlm_extractor import extract_clip_attributes
from src.utils.config import get_settings

CLIP_ID = "clip_test_m5"
VIDEO_ID = "video_test_m5"


class FakeBackend:
    """Records every caption() call. Frame 0's crop "succeeds" on the primary
    task; frame 1's crop only yields a match on the retry task — this is what
    exercises the retry path."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []  # (n_crops, task)
        self.PRIMARY_TASK, self.RETRY_TASK = vlm_extractor.BACKEND_TASKS["florence2"]

    def caption(self, crops_bgr, task):
        self.calls.append((len(crops_bgr), task))
        captions = []
        for crop in crops_bgr:
            # Distinguish the two fixture crops by their fill value.
            is_frame_0 = bool((crop == 10).all())
            if task == self.PRIMARY_TASK:
                captions.append("a white sedan" if is_frame_0 else "an indescribable object")
            else:
                captions.append("should not be reached" if is_frame_0 else "a blue truck")
        return captions


@pytest.fixture
def fixture_paths():
    settings = get_settings()
    frames_dir = settings.resolve_path(settings.paths.frames_dir) / CLIP_ID
    ingest_dir = settings.resolve_path(settings.paths.outputs_dir) / "ingest"
    tracks_dir = settings.resolve_path(settings.paths.outputs_dir) / "tracks_associated"
    attrs_dir = settings.resolve_path(settings.paths.outputs_dir) / "attributes_raw"
    cache_dir = settings.resolve_path(settings.vlm.cache_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    frame0 = np.full((40, 40, 3), 10, dtype=np.uint8)
    frame1 = np.full((40, 40, 3), 20, dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000000.jpg"), frame0)
    cv2.imwrite(str(frames_dir / "frame_000001.jpg"), frame1)

    manifest = {
        "video_id": VIDEO_ID,
        "source_path": "fixture.mp4",
        "fps": 25.0,
        "resolution": [40, 40],
        "frame_count": 2,
        "start_wallclock": "2026-08-15T10:00:00",
        "end_wallclock": "2026-08-15T10:00:00.08",
        "clips": [
            {
                "video_id": VIDEO_ID,
                "clip_id": CLIP_ID,
                "clip_path": "unused.mp4",
                "start_wallclock": "2026-08-15T10:00:00",
                "end_wallclock": "2026-08-15T10:00:00.08",
                "fps": 25.0,
                "resolution": [40, 40],
                "frames": [
                    {
                        "clip_id": CLIP_ID, "frame_id": "frame_000000",
                        "frame_path": f"data/frames/{CLIP_ID}/frame_000000.jpg",
                        "video_timestamp_sec": 0.0, "wallclock_time": "2026-08-15T10:00:00",
                        "fps": 25.0, "resolution": [40, 40],
                    },
                    {
                        "clip_id": CLIP_ID, "frame_id": "frame_000001",
                        "frame_path": f"data/frames/{CLIP_ID}/frame_000001.jpg",
                        "video_timestamp_sec": 0.04, "wallclock_time": "2026-08-15T10:00:00.04",
                        "fps": 25.0, "resolution": [40, 40],
                    },
                ],
            }
        ],
    }
    (ingest_dir / f"{VIDEO_ID}_manifest.json").write_text(json.dumps(manifest))

    associated = {
        "clip_id": CLIP_ID,
        "tracks": [
            {
                "track_id": "1", "class": "car",
                "frames": ["frame_000000"], "bboxes": [[0.0, 0.0, 40.0, 40.0]],
                "centers": [[20.0, 20.0]], "velocity": [[0.0, 0.0]],
                "dominant_direction_deg": None, "embedding": [],
                "best_shot_crops": [], "best_shot_scores": [],
            },
            {
                "track_id": "2", "class": "car",
                "frames": ["frame_000001"], "bboxes": [[0.0, 0.0, 40.0, 40.0]],
                "centers": [[20.0, 20.0]], "velocity": [[0.0, 0.0]],
                "dominant_direction_deg": None, "embedding": [],
                "best_shot_crops": [], "best_shot_scores": [],
            },
        ],
        "merge_log": [], "rejected_links": [], "suppressed_links": [],
    }
    (tracks_dir / f"{CLIP_ID}.json").write_text(json.dumps(associated))

    yield settings

    shutil.rmtree(frames_dir, ignore_errors=True)
    (ingest_dir / f"{VIDEO_ID}_manifest.json").unlink(missing_ok=True)
    (tracks_dir / f"{CLIP_ID}.json").unlink(missing_ok=True)
    (attrs_dir / f"{CLIP_ID}.json").unlink(missing_ok=True)
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_retry_only_fires_for_crops_that_needed_it(fixture_paths, monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(vlm_extractor, "_load_backend", lambda settings: fake)

    result = extract_clip_attributes(CLIP_ID, settings=fixture_paths)

    by_track = {a.track_id: a for a in result.attributes}
    assert by_track["1"].color == "white"
    assert by_track["1"].vehicle_type == "sedan"
    assert by_track["2"].color == "blue"
    assert by_track["2"].vehicle_type == "truck"

    # One primary-task batch call, one retry-task call for exactly the crop
    # that needed it (frame 1) — never the one that already succeeded.
    primary_calls = [c for c in fake.calls if c[1] == fake.PRIMARY_TASK]
    retry_calls = [c for c in fake.calls if c[1] == fake.RETRY_TASK]
    assert sum(n for n, _ in primary_calls) == 2
    assert retry_calls == [(1, fake.RETRY_TASK)]


def test_second_run_hits_cache_and_never_touches_backend(fixture_paths, monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(vlm_extractor, "_load_backend", lambda settings: fake)
    extract_clip_attributes(CLIP_ID, settings=fixture_paths)
    assert len(fake.calls) > 0

    def _fail(_settings):
        raise AssertionError("backend must not be loaded when every crop is cached")

    monkeypatch.setattr(vlm_extractor, "_load_backend", _fail)
    result = extract_clip_attributes(CLIP_ID, settings=fixture_paths)
    assert len(result.attributes) == 2
