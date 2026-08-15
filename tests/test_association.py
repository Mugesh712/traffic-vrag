"""Gate-level tests for M4 association.

Each hard gate is exercised in isolation: a baseline pair that passes every
gate, then one perturbation per gate that must flip it to a rejection.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.perception.association import _evaluate_gates, _merge_tracks, _summarize
from src.utils.config import get_settings
from src.utils.schemas import Track

# Sampled frames, half a second apart.
FRAMES = [f"f{i}" for i in range(8)]
POSITIONS = {f: i for i, f in enumerate(FRAMES)}
TIMESTAMPS = {f: 0.5 * i for i, f in enumerate(FRAMES)}

EMB_A = [1.0, 0.0, 0.0, 0.0]
EMB_ORTHOGONAL = [0.0, 1.0, 0.0, 0.0]


def make_track(
    track_id: str,
    frame_ids: list[str],
    centers: list[tuple[float, float]],
    embedding: list[float] = EMB_A,
    cls: str = "car",
    half: float = 25.0,
) -> Track:
    bboxes = [(x - half, y - half, x + half, y + half) for x, y in centers]
    velocity: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(1, len(centers)):
        dt = TIMESTAMPS[frame_ids[i]] - TIMESTAMPS[frame_ids[i - 1]]
        velocity.append(((centers[i][0] - centers[i - 1][0]) / dt, (centers[i][1] - centers[i - 1][1]) / dt))
    return Track(
        track_id=track_id,
        **{"class": cls},
        frames=frame_ids,
        bboxes=bboxes,
        centers=centers,
        velocity=velocity,
        embedding=embedding,
        best_shot_crops=[f"crop_{track_id}.jpg"],
        best_shot_scores=[1.0],
    )


def summarize(track: Track):
    return _summarize(track, POSITIONS, TIMESTAMPS, get_settings())


def gates(a: Track, b: Track):
    return _evaluate_gates(summarize(a), summarize(b), get_settings())


# A moves right at 100 px/s and stops at f2; B resumes at f4 exactly where a
# constant-velocity extrapolation says it should be.
LEADER = make_track("1", FRAMES[0:3], [(100.0, 100.0), (150.0, 100.0), (200.0, 100.0)])


def test_baseline_pair_passes_every_gate():
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)])
    failed, scores = gates(LEADER, follower)
    assert failed is None
    assert scores["temporal_gap_sec"] == pytest.approx(1.0)
    assert scores["motion_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["appearance"] == pytest.approx(1.0)


def test_class_gate_rejects_class_change():
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)], cls="truck")
    failed, _ = gates(LEADER, follower)
    assert failed == "class"


def test_temporal_gate_rejects_overlap():
    """Two tracks sharing frames cannot be one object in two places."""
    follower = make_track("2", FRAMES[2:4], [(200.0, 100.0), (250.0, 100.0)])
    failed, _ = gates(LEADER, follower)
    assert failed == "temporal"


def test_temporal_gate_rejects_gap_beyond_budget(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.association, "max_gap_sec", 0.4)
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)])
    failed, _ = _evaluate_gates(summarize(LEADER), summarize(follower), settings)
    assert failed == "temporal"


def test_motion_gate_rejects_implausible_jump():
    """Same appearance and class, but nowhere near where physics puts it."""
    follower = make_track("2", FRAMES[4:6], [(300.0, 900.0), (350.0, 900.0)])
    failed, scores = gates(LEADER, follower)
    assert failed == "motion"
    assert scores["motion_error"] > get_settings().association.motion_tolerance


def test_appearance_gate_vetoes_motion_plausible_impostor():
    """The decoy case: geometry says yes, appearance says no."""
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)], embedding=EMB_ORTHOGONAL)
    failed, scores = gates(LEADER, follower)
    assert failed == "appearance"
    assert scores["appearance"] == pytest.approx(0.0, abs=1e-9)


def test_stationary_pair_matches_on_iou_not_prediction():
    """A parked car has no velocity to extrapolate, so overlap decides."""
    parked_a = make_track("1", FRAMES[0:3], [(100.0, 100.0), (100.5, 100.0), (100.0, 100.5)])
    parked_b = make_track("2", FRAMES[4:6], [(101.0, 100.0), (100.5, 100.0)])
    failed, scores = gates(parked_a, parked_b)
    assert failed is None
    assert scores["motion_iou"] > 0.9  # IoU path, not the prediction path
    assert "motion_error" not in scores


def test_stationary_pair_rejected_when_boxes_do_not_overlap():
    parked_a = make_track("1", FRAMES[0:3], [(100.0, 100.0), (100.5, 100.0), (100.0, 100.5)])
    parked_far = make_track("2", FRAMES[4:6], [(600.0, 600.0), (600.5, 600.0)])
    failed, scores = gates(parked_a, parked_far)
    assert failed == "motion"
    assert scores["motion_iou"] == 0.0


def test_merge_is_independent_of_member_order():
    """Determinism: the merged track must not depend on input ordering."""
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)])
    members = [summarize(LEADER), summarize(follower)]
    forward = _merge_tracks(members, TIMESTAMPS, top_k=5)
    reverse = _merge_tracks(list(reversed(members)), TIMESTAMPS, top_k=5)
    assert forward.model_dump() == reverse.model_dump()
    assert forward.track_id == "1"  # earliest fragment's id survives
    assert forward.frames == FRAMES[0:3] + FRAMES[4:6]


def test_merged_velocity_spans_the_repaired_gap():
    """The frame after a repaired gap gets a real velocity, not a placeholder."""
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)])
    merged = _merge_tracks([summarize(LEADER), summarize(follower)], TIMESTAMPS, top_k=5)
    gap_velocity = merged.velocity[3]  # f2 -> f4, 100 px over 1.0 s
    assert gap_velocity[0] == pytest.approx(100.0)
    assert merged.dominant_direction_deg == pytest.approx(0.0)


def test_merged_embedding_is_frame_count_weighted_and_normalized():
    follower = make_track("2", FRAMES[4:6], [(300.0, 100.0), (350.0, 100.0)], embedding=EMB_ORTHOGONAL)
    merged = _merge_tracks([summarize(LEADER), summarize(follower)], TIMESTAMPS, top_k=5)
    embedding = np.asarray(merged.embedding)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    # 3 frames of EMB_A against 2 of the orthogonal one.
    assert embedding[0] > embedding[1]
