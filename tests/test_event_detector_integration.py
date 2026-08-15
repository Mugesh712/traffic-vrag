"""End-to-end test for M9's orchestration: trajectory stitching across a real
clip boundary, event_id assignment, and regions-file loading.

Fixtures written under the real data/ tree and cleaned up unconditionally.
"""
from __future__ import annotations

import json
import shutil

import pytest
import yaml

from src.graph.event_detector import detect_events
from src.utils.config import get_settings

VIDEO_ID = "video_test_m9"
CLIP_A, CLIP_B = "clip_test_m9_a", "clip_test_m9_b"


def frame(clip_id: str, frame_id: str, ts: float) -> dict:
    return {
        "clip_id": clip_id, "frame_id": frame_id,
        "frame_path": f"data/frames/unused/{frame_id}.jpg",
        "video_timestamp_sec": ts, "wallclock_time": f"2026-08-15T10:00:{ts:05.2f}",
        "fps": 25.0, "resolution": [640, 480],
    }


def track(track_id: str, frames: list[str], centers: list[tuple[float, float]]) -> dict:
    return {
        "track_id": track_id, "class": "car", "frames": frames,
        "bboxes": [[c[0] - 25, c[1] - 25, c[0] + 25, c[1] + 25] for c in centers],
        "centers": [list(c) for c in centers],
        "velocity": [[0.0, 0.0]] + [
            [(centers[i][0] - centers[i - 1][0]) * 25, (centers[i][1] - centers[i - 1][1]) * 25]
            for i in range(1, len(centers))
        ],
        "dominant_direction_deg": None, "embedding": [],
        "best_shot_crops": [], "best_shot_scores": [],
    }


@pytest.fixture
def settings():
    cfg = get_settings()
    yield cfg
    outputs = cfg.resolve_path(cfg.paths.outputs_dir)
    (outputs / "ingest" / f"{VIDEO_ID}_manifest.json").unlink(missing_ok=True)
    for clip_id in (CLIP_A, CLIP_B):
        (outputs / "tracks_associated" / f"{clip_id}.json").unlink(missing_ok=True)
    (outputs / "master_object_index" / f"{VIDEO_ID}.json").unlink(missing_ok=True)
    (outputs / "events" / f"{VIDEO_ID}.json").unlink(missing_ok=True)
    (cfg.resolve_path(cfg.events.regions_dir) / f"{VIDEO_ID}.yaml").unlink(missing_ok=True)


def write_fixture(settings, with_regions: bool = False):
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    for sub in ("ingest", "tracks_associated", "master_object_index"):
        (outputs / sub).mkdir(parents=True, exist_ok=True)

    # One object, moving east at 100 px/s, split across two clips at t=1.0s.
    # Frame sample interval is 0.04s (25fps) so the trajectories interleave
    # tightly across the boundary -- if global timestamps were reset per clip,
    # this would collapse into two overlapping (broken) segments instead of
    # one continuous line.
    frames_a = ["f0", "f1"]
    ts_a = [0.0, 1.0]
    centers_a = [(0.0, 50.0), (100.0, 50.0)]

    frames_b = ["f2", "f3"]
    ts_b = [1.04, 2.0]
    centers_b = [(104.0, 50.0), (200.0, 50.0)]

    manifest = {
        "video_id": VIDEO_ID, "source_path": "fixture.mp4", "fps": 25.0,
        "resolution": [640, 480], "frame_count": 4,
        "start_wallclock": "2026-08-15T10:00:00", "end_wallclock": "2026-08-15T10:00:02",
        "clips": [
            {
                "video_id": VIDEO_ID, "clip_id": CLIP_A, "clip_path": "a.mp4",
                "start_wallclock": "2026-08-15T10:00:00", "end_wallclock": "2026-08-15T10:00:01",
                "fps": 25.0, "resolution": [640, 480],
                "frames": [frame(CLIP_A, f, t) for f, t in zip(frames_a, ts_a)],
            },
            {
                "video_id": VIDEO_ID, "clip_id": CLIP_B, "clip_path": "b.mp4",
                "start_wallclock": "2026-08-15T10:00:01", "end_wallclock": "2026-08-15T10:00:02",
                "fps": 25.0, "resolution": [640, 480],
                "frames": [frame(CLIP_B, f, t) for f, t in zip(frames_b, ts_b)],
            },
        ],
    }
    (outputs / "ingest" / f"{VIDEO_ID}_manifest.json").write_text(json.dumps(manifest))

    (outputs / "tracks_associated" / f"{CLIP_A}.json").write_text(
        json.dumps(
            {"clip_id": CLIP_A, "tracks": [track("1", frames_a, centers_a)],
             "merge_log": [], "rejected_links": [], "suppressed_links": []}
        )
    )
    (outputs / "tracks_associated" / f"{CLIP_B}.json").write_text(
        json.dumps(
            {"clip_id": CLIP_B, "tracks": [track("7", frames_b, centers_b)],
             "merge_log": [], "rejected_links": [], "suppressed_links": []}
        )
    )

    (outputs / "master_object_index" / f"{VIDEO_ID}.json").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "objects": [
                    {
                        "global_id": "obj_0001", "class": "car",
                        "sightings": [
                            {"clip_id": CLIP_A, "track_id": "1", "first_seen": "x", "last_seen": "x", "gate_scores": {}},
                            {"clip_id": CLIP_B, "track_id": "7", "first_seen": "x", "last_seen": "x", "gate_scores": {}},
                        ],
                    }
                ],
                "rejected_links": [], "suppressed_links": [],
            }
        )
    )

    if with_regions:
        # Sits strictly between A's last sample (x=100) and B's first (x=104),
        # so the only possible crossing is the boundary-spanning segment.
        regions_dir = settings.resolve_path(settings.events.regions_dir)
        regions_dir.mkdir(parents=True, exist_ok=True)
        (regions_dir / f"{VIDEO_ID}.yaml").write_text(
            yaml.dump({"lines": [{"id": "midline", "points": [[102, 0], [102, 100]]}]})
        )


def test_trajectory_stitches_across_clip_boundary_on_the_global_time_axis(settings):
    """No offset arithmetic needed: M1's video_timestamp_sec is already global,
    so clip A's last sample (t=1.0) and clip B's first (t=1.04) interleave
    correctly rather than both starting near t=0."""
    write_fixture(settings)
    log = detect_events(VIDEO_ID, settings=settings)
    # A straight, steady drive shouldn't itself produce spurious events; the
    # real assertion is in the CROSSES test below, which depends on the
    # stitched trajectory actually being continuous and monotonic in time.
    assert all(e.subject_id == "obj_0001" for e in log.events)


def test_crosses_a_region_line_spanning_the_clip_boundary(settings):
    """The crossing happens between clip A's last sample and clip B's first --
    proof the two clips' tracks were joined into one continuous trajectory."""
    write_fixture(settings, with_regions=True)
    log = detect_events(VIDEO_ID, settings=settings)
    crosses = [e for e in log.events if e.type == "CROSSES"]
    assert len(crosses) == 1
    assert crosses[0].metadata["region_id"] == "midline"
    assert crosses[0].start_time == "2026-08-15T10:00:01.00"
    assert crosses[0].end_time == "2026-08-15T10:00:01.04"


def test_no_regions_file_means_no_crosses_events_not_an_error(settings):
    write_fixture(settings, with_regions=False)
    log = detect_events(VIDEO_ID, settings=settings)
    assert [e for e in log.events if e.type == "CROSSES"] == []


def test_event_ids_are_sequential_and_deterministic(settings):
    write_fixture(settings, with_regions=True)
    log = detect_events(VIDEO_ID, settings=settings)
    ids = [e.event_id for e in log.events]
    assert ids == [f"evt_{n:05d}" for n in range(1, len(ids) + 1)]

    first_run = json.loads(
        (settings.resolve_path(settings.paths.outputs_dir) / "events" / f"{VIDEO_ID}.json").read_text()
    )
    log2 = detect_events(VIDEO_ID, settings=settings)
    assert [e.model_dump() for e in log.events] == [e.model_dump() for e in log2.events]
