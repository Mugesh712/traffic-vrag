"""Gate-level tests for M7 cross-clip global linking."""
from __future__ import annotations

import numpy as np
import pytest

from src.semantics.global_linking import (
    SEMANTIC_ATTRIBUTES,
    _ClipTrackSummary,
    _combined_score,
    _evaluate_gates,
)
from src.utils.config import get_settings

EMB_A = np.array([1.0, 0.0, 0.0, 0.0])
EMB_ORTHOGONAL = np.array([0.0, 1.0, 0.0, 0.0])


def summary(
    clip_id: str,
    track_id: str,
    *,
    cls: str = "car",
    first_ts: float = 0.0,
    last_ts: float = 1.0,
    entry_center: tuple[float, float] = (100.0, 100.0),
    exit_center: tuple[float, float] = (200.0, 100.0),
    exit_velocity: tuple[float, float] = (100.0, 0.0),
    embedding: np.ndarray = EMB_A,
    attributes: dict | None = None,
) -> _ClipTrackSummary:
    return _ClipTrackSummary(
        clip_id=clip_id,
        track_id=track_id,
        cls=cls,
        first_ts=first_ts,
        last_ts=last_ts,
        first_wallclock="2026-08-15T10:00:00",
        last_wallclock="2026-08-15T10:00:01",
        entry_center=entry_center,
        exit_center=exit_center,
        exit_bbox=(175.0, 75.0, 225.0, 125.0),  # 50x50, diagonal ~70.7
        exit_velocity=exit_velocity,
        embedding=embedding,
        attributes=attributes or {},
    )


def certain(**values) -> dict:
    return {attr: (value, False) for attr, value in values.items()}


def gates(a, b, settings=None):
    return _evaluate_gates(a, b, settings or get_settings())


# Leaves clip A at (200,100) moving right at 100 px/s; B enters 1s later at
# exactly where a constant-velocity prediction puts it.
LEADER = summary("clip_000", "1", last_ts=10.0, attributes=certain(color="white", vehicle_type="sedan"))
FOLLOWER = summary(
    "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
    attributes=certain(color="white", vehicle_type="sedan"),
)


def test_baseline_link_passes_every_gate():
    failed, scores = gates(LEADER, FOLLOWER)
    assert failed is None
    assert scores["gap_sec"] == pytest.approx(1.0)
    assert scores["motion_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["appearance"] == pytest.approx(1.0)
    assert scores["semantic"] == pytest.approx(1.0)


def test_class_gate_rejects():
    failed, _ = gates(LEADER, summary("clip_001", "5", cls="truck", first_ts=11.0, entry_center=(300.0, 100.0)))
    assert failed == "class"


# --- the semantic gate ----------------------------------------------------


def test_semantic_gate_rejects_confident_disagreement():
    """A white sedan cannot become a blue sedan, however well motion lines up."""
    impostor = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
        attributes=certain(color="blue", vehicle_type="sedan"),
    )
    failed, scores = gates(LEADER, impostor)
    assert failed == "semantic"
    assert scores["semantic_comparable"] == 2.0
    assert scores["semantic_agreed"] == 1.0


def test_uncertain_attribute_matches_anything():
    """M6's uncertainty flag paying for M7: an honest 'unsure' must not veto."""
    unsure = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
        attributes={"color": ("blue", True), "vehicle_type": ("sedan", False)},
    )
    failed, scores = gates(LEADER, unsure)
    assert failed is None
    assert scores["semantic_comparable"] == 1.0  # colour skipped, type compared


def test_missing_attributes_are_treated_as_uncertain():
    bare = summary("clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0))
    failed, scores = gates(LEADER, bare)
    assert failed is None
    assert scores["semantic"] == pytest.approx(1.0)
    assert scores["semantic_comparable"] == 0.0


def test_direction_is_state_not_identity_and_never_gates():
    """A car that turns a corner changes direction legitimately."""
    assert "direction" not in SEMANTIC_ATTRIBUTES
    turned = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
        attributes=certain(color="white", vehicle_type="sedan", direction="left"),
    )
    a = summary(
        "clip_000", "1", last_ts=10.0,
        attributes=certain(color="white", vehicle_type="sedan", direction="right"),
    )
    failed, _ = gates(a, turned)
    assert failed is None


def test_semantic_gate_can_be_disabled_for_ablation(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.linking, "enable_semantic_gate", False)
    impostor = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
        attributes=certain(color="blue", vehicle_type="truck"),
    )
    failed, _ = gates(LEADER, impostor, settings)
    assert failed is None  # would have been "semantic" with the gate on


# --- motion and appearance ------------------------------------------------


def test_motion_gate_rejects_implausible_entry_position():
    far = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 900.0),
        attributes=certain(color="white", vehicle_type="sedan"),
    )
    failed, scores = gates(LEADER, far)
    assert failed == "motion"
    assert scores["motion_error"] > get_settings().linking.motion_tolerance


def test_motion_gate_rejects_stale_track_from_mid_clip():
    """The reason max_gap_sec is small: a track that vanished long before the
    boundary must not be linked across it."""
    stale = summary("clip_000", "9", last_ts=0.0, attributes=certain(color="white"))
    failed, scores = gates(stale, FOLLOWER)
    assert failed == "motion"
    assert scores["gap_sec"] > get_settings().linking.max_gap_sec


def test_motion_gate_rejects_backwards_in_time():
    failed, _ = gates(FOLLOWER, LEADER)
    assert failed == "motion"


def test_appearance_gate_vetoes_lookalike_trajectory():
    impostor = summary(
        "clip_001", "5", first_ts=11.0, entry_center=(300.0, 100.0),
        embedding=EMB_ORTHOGONAL, attributes=certain(color="white", vehicle_type="sedan"),
    )
    failed, scores = gates(LEADER, impostor)
    assert failed == "appearance"
    assert scores["appearance"] == pytest.approx(0.0, abs=1e-9)


def test_combined_score_is_weighted_sum():
    _, scores = gates(LEADER, FOLLOWER)
    settings = get_settings()
    w = settings.linking.score_weights
    assert _combined_score(scores, settings) == pytest.approx(w.appearance + w.motion + w.semantic)
