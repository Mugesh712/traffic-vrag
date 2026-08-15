"""End-to-end tests for M7's linking pass over a whole video.

Writes fixtures under the real data/ tree (resolve_path is anchored to the
project root) and cleans up unconditionally.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.semantics.global_linking import link_video
from src.utils.config import get_settings

VIDEO_ID = "video_test_m7"
CLIP_A, CLIP_B = "clip_test_m7_a", "clip_test_m7_b"

# Geometry shared by every fixture track: each clip-A track exits at (100,100)
# stationary, each clip-B track enters at (110,100). Motion error is therefore
# identical for all four pairs, so the assignment is decided purely by
# appearance -- which is what these tests are about.
EXIT_BBOX = [75.0, 75.0, 125.0, 125.0]
ENTRY_BBOX = [85.0, 75.0, 135.0, 125.0]
EXIT_CENTER = [100.0, 100.0]
ENTRY_CENTER = [110.0, 100.0]


def unit(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def track(track_id: str, frames: list[str], centers, bbox, embedding, cls: str = "car") -> dict:
    return {
        "track_id": track_id,
        "class": cls,
        "frames": frames,
        "bboxes": [list(bbox) for _ in frames],
        "centers": [list(centers) for _ in frames],
        "velocity": [[0.0, 0.0] for _ in frames],
        "dominant_direction_deg": None,
        "embedding": list(embedding),
        "best_shot_crops": [],
        "best_shot_scores": [],
    }


def write_fixture(clip_a_tracks: list[dict], clip_b_tracks: list[dict], settings) -> None:
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    (outputs / "ingest").mkdir(parents=True, exist_ok=True)
    (outputs / "tracks_associated").mkdir(parents=True, exist_ok=True)

    def frame(frame_id: str, ts: float) -> dict:
        return {
            "clip_id": CLIP_A if frame_id in ("f0", "f1") else CLIP_B,
            "frame_id": frame_id,
            "frame_path": f"data/frames/unused/{frame_id}.jpg",
            "video_timestamp_sec": ts,
            "wallclock_time": f"2026-08-15T10:00:0{int(ts)}",
            "fps": 25.0,
            "resolution": [640, 480],
        }

    manifest = {
        "video_id": VIDEO_ID,
        "source_path": "fixture.mp4",
        "fps": 25.0,
        "resolution": [640, 480],
        "frame_count": 4,
        "start_wallclock": "2026-08-15T10:00:00",
        "end_wallclock": "2026-08-15T10:00:02",
        "clips": [
            {
                "video_id": VIDEO_ID, "clip_id": CLIP_A, "clip_path": "a.mp4",
                "start_wallclock": "2026-08-15T10:00:00", "end_wallclock": "2026-08-15T10:00:01",
                "fps": 25.0, "resolution": [640, 480],
                "frames": [frame("f0", 0.0), frame("f1", 1.0)],
            },
            {
                "video_id": VIDEO_ID, "clip_id": CLIP_B, "clip_path": "b.mp4",
                "start_wallclock": "2026-08-15T10:00:01", "end_wallclock": "2026-08-15T10:00:02",
                "fps": 25.0, "resolution": [640, 480],
                "frames": [frame("f2", 1.5), frame("f3", 2.0)],
            },
        ],
    }
    (outputs / "ingest" / f"{VIDEO_ID}_manifest.json").write_text(json.dumps(manifest))

    for clip_id, tracks in ((CLIP_A, clip_a_tracks), (CLIP_B, clip_b_tracks)):
        (outputs / "tracks_associated" / f"{clip_id}.json").write_text(
            json.dumps(
                {
                    "clip_id": clip_id, "tracks": tracks,
                    "merge_log": [], "rejected_links": [], "suppressed_links": [],
                }
            )
        )


@pytest.fixture
def settings():
    cfg = get_settings()
    yield cfg
    outputs = cfg.resolve_path(cfg.paths.outputs_dir)
    (outputs / "ingest" / f"{VIDEO_ID}_manifest.json").unlink(missing_ok=True)
    for clip_id in (CLIP_A, CLIP_B):
        (outputs / "tracks_associated" / f"{clip_id}.json").unlink(missing_ok=True)
        (outputs / "attributes_canonical" / f"{clip_id}.json").unlink(missing_ok=True)
    (outputs / "master_object_index" / f"{VIDEO_ID}.json").unlink(missing_ok=True)


def test_matching_tracks_across_a_boundary_become_one_object(settings):
    write_fixture(
        [track("1", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, unit(0))],
        [track("5", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, unit(0))],
        settings,
    )
    index = link_video(VIDEO_ID, settings=settings)

    assert len(index.objects) == 1
    obj = index.objects[0]
    assert obj.global_id == "obj_0001"
    assert [(s.clip_id, s.track_id) for s in obj.sightings] == [(CLIP_A, "1"), (CLIP_B, "5")]
    # The first sighting has no incoming link; the second records the gates.
    assert obj.sightings[0].gate_scores == {}
    assert obj.sightings[1].gate_scores["appearance"] == pytest.approx(1.0)


def test_unlinkable_tracks_stay_separate_objects(settings):
    write_fixture(
        [track("1", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, unit(0))],
        [track("5", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, unit(0), cls="truck")],
        settings,
    )
    index = link_video(VIDEO_ID, settings=settings)

    assert len(index.objects) == 2
    assert all(len(o.sightings) == 1 for o in index.objects)
    assert [r.failed_gate for r in index.rejected_links] == ["class"]


def test_assignment_is_global_not_greedy(settings):
    """The reason M7 uses Hungarian rather than best-first matching.

    Cosine matrix (rows clip-A, cols clip-B):

                b1      b2
        a1    0.990   0.800
        a2    0.980   0.763

    Greedy takes the single best cell (a1-b1, 0.990) and is then forced onto
    a2-b2 (0.763), totalling 1.753. The optimal assignment is a1-b2 + a2-b1,
    totalling 1.780. Motion and semantics are identical across all four pairs,
    so appearance alone decides and the two strategies must disagree.
    """
    a1, a2 = unit(0), unit(-3.37)
    b1, b2 = unit(8.11), unit(36.87)

    # Confirm the fixture really does separate the two strategies.
    cos = np.array([[np.dot(a, b) for b in (b1, b2)] for a in (a1, a2)])
    assert cos.min() > settings.linking.appearance_similarity_threshold  # all four admissible
    greedy_total = cos[0, 0] + cos[1, 1]
    optimal_total = cos[0, 1] + cos[1, 0]
    assert optimal_total > greedy_total

    write_fixture(
        [
            track("1", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, a1),
            track("2", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, a2),
        ],
        [
            track("5", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, b1),
            track("6", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, b2),
        ],
        settings,
    )
    index = link_video(VIDEO_ID, settings=settings)

    assert len(index.objects) == 2
    pairs = {
        next(s.track_id for s in o.sightings if s.clip_id == CLIP_A):
        next(s.track_id for s in o.sightings if s.clip_id == CLIP_B)
        for o in index.objects
    }
    assert pairs == {"1": "6", "2": "5"}, "solver fell back to greedy best-first"

    # The two admissible-but-unchosen pairs are logged, not silently dropped.
    assert len(index.suppressed_links) == 2
    assert {r.failed_gate for r in index.suppressed_links} == {"assignment_conflict"}


def test_each_track_keeps_at_most_one_successor(settings):
    """Two clip-A tracks cannot both claim the same clip-B track."""
    write_fixture(
        [
            track("1", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, unit(0)),
            track("2", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, unit(1)),
        ],
        [track("5", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, unit(0))],
        settings,
    )
    index = link_video(VIDEO_ID, settings=settings)

    assert len(index.objects) == 2  # one linked pair, one orphan
    assert sorted(len(o.sightings) for o in index.objects) == [1, 2]


def test_uncertain_attributes_do_not_block_a_correct_link(settings):
    """M6's uncertainty flag composing with M7's semantic gate: a value the
    system was never sure about must not veto a link the geometry supports."""
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    (outputs / "attributes_canonical").mkdir(parents=True, exist_ok=True)

    def canonical(clip_id: str, track_id: str, color: str, uncertain: bool) -> None:
        (outputs / "attributes_canonical" / f"{clip_id}.json").write_text(
            json.dumps(
                {
                    "clip_id": clip_id,
                    "tracks": [
                        {
                            "track_id": track_id,
                            "votes": [
                                {
                                    "attribute": "color", "winner": color,
                                    "distribution": {color: 1.0}, "uncertain": uncertain,
                                    "n_votes": 3, "margin": 1.0,
                                    "uncertain_reason": "low_margin" if uncertain else None,
                                }
                            ],
                        }
                    ],
                }
            )
        )

    write_fixture(
        [track("1", ["f0", "f1"], EXIT_CENTER, EXIT_BBOX, unit(0))],
        [track("5", ["f2", "f3"], ENTRY_CENTER, ENTRY_BBOX, unit(0))],
        settings,
    )
    # Confidently white, then an unsure "blue": the disagreement is real but
    # the system does not stand behind it, so it must not gate.
    canonical(CLIP_A, "1", "white", uncertain=False)
    canonical(CLIP_B, "5", "blue", uncertain=True)

    index = link_video(VIDEO_ID, settings=settings)
    assert len(index.objects) == 1, "an uncertain attribute wrongly vetoed the link"

    # Same fixture, but now the system stands behind the disagreement.
    canonical(CLIP_B, "5", "blue", uncertain=False)
    index = link_video(VIDEO_ID, settings=settings)
    assert len(index.objects) == 2
    assert [r.failed_gate for r in index.rejected_links] == ["semantic"]
