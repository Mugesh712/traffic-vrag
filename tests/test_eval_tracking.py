"""Tests for M16's tracking metrics, against hand-computable known answers.

Every expected value here is derived from the metric definitions by hand, not
from the implementation's own output -- a test that just records what the code
currently does would pass even if the formula were wrong.
"""
from __future__ import annotations

import pytest

from src.eval.tracking import FrameBoxes, evaluate_tracking, iou, match_frame

BOX_A = [0.0, 0.0, 10.0, 10.0]
BOX_A_SHIFTED = [1.0, 0.0, 11.0, 10.0]  # IoU with BOX_A ~= 0.818
BOX_B = [100.0, 100.0, 110.0, 110.0]


def frames(*per_frame: dict[str, list[float]]) -> list[FrameBoxes]:
    return [FrameBoxes(frame_id=f"f{i}", boxes=b) for i, b in enumerate(per_frame)]


# --- primitives -------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert iou(BOX_A, BOX_A) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou(BOX_A, BOX_B) == 0.0


def test_iou_half_overlap():
    # 10x10 and a 10x10 shifted 5 right: intersection 50, union 150.
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_match_frame_pairs_by_best_overlap():
    gt = {"g1": BOX_A, "g2": BOX_B}
    pred = {"p1": BOX_B, "p2": BOX_A_SHIFTED}
    assert sorted(match_frame(gt, pred, 0.5)) == [("g1", "p2"), ("g2", "p1")]


def test_match_frame_rejects_pairs_below_threshold():
    assert match_frame({"g1": BOX_A}, {"p1": BOX_B}, 0.5) == []


def test_match_frame_handles_empty_sides():
    assert match_frame({}, {"p1": BOX_A}, 0.5) == []
    assert match_frame({"g1": BOX_A}, {}, 0.5) == []


# --- perfect tracker --------------------------------------------------------


def test_perfect_tracking_scores_one():
    gt = frames({"g1": BOX_A}, {"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {"p1": BOX_A}, {"p1": BOX_A})
    m = evaluate_tracking(gt, pred)
    assert m.mota == pytest.approx(1.0)
    assert m.idf1 == pytest.approx(1.0)
    assert m.id_switches == 0
    assert m.false_positives == 0
    assert m.false_negatives == 0


# --- ID switch --------------------------------------------------------------


def test_single_id_switch_is_counted_once():
    """One GT identity, tracked correctly, but the predicted id changes at
    frame 2. 3 GT boxes, 0 FP, 0 FN, 1 IDSW -> MOTA = 1 - 1/3."""
    gt = frames({"g1": BOX_A}, {"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {"p2": BOX_A}, {"p2": BOX_A})
    m = evaluate_tracking(gt, pred)
    assert m.id_switches == 1
    assert m.mota == pytest.approx(1 - 1 / 3)


def test_idf1_punishes_fragmentation_harder_than_mota():
    """The reason both are reported. One GT identity of 4 boxes, split evenly
    across two predicted ids.

    MOTA sees a single ID switch:            1 - 1/4 = 0.75
    IDF1 solves a global assignment, so only one predicted id can claim the
    identity: IDTP=2, IDFP=2, IDFN=2 -> 2*2/(2*2+2+2) = 0.5
    """
    gt = frames({"g1": BOX_A}, {"g1": BOX_A}, {"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {"p1": BOX_A}, {"p2": BOX_A}, {"p2": BOX_A})
    m = evaluate_tracking(gt, pred)
    assert m.mota == pytest.approx(0.75)
    assert m.idf1 == pytest.approx(0.5)
    assert m.idf1 < m.mota


# --- misses and false positives ---------------------------------------------


def test_missed_detection_is_a_false_negative():
    gt = frames({"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {})
    m = evaluate_tracking(gt, pred)
    assert m.false_negatives == 1
    assert m.mota == pytest.approx(1 - 1 / 2)


def test_spurious_box_is_a_false_positive():
    gt = frames({"g1": BOX_A})
    pred = frames({"p1": BOX_A, "p2": BOX_B})
    m = evaluate_tracking(gt, pred)
    assert m.false_positives == 1
    assert m.mota == pytest.approx(0.0)  # 1 - 1/1


def test_mota_can_go_negative_and_is_not_clamped():
    """A tracker emitting many false positives genuinely scores below zero;
    clamping would hide exactly the failure worth reporting."""
    gt = frames({"g1": BOX_A})
    pred = frames({"p1": BOX_A, "p2": BOX_B, "p3": [200, 200, 210, 210], "p4": [300, 300, 310, 310]})
    m = evaluate_tracking(gt, pred)
    assert m.false_positives == 3
    assert m.mota == pytest.approx(-2.0)


def test_empty_prediction_scores_zero_not_an_error():
    gt = frames({"g1": BOX_A}, {"g1": BOX_A})
    m = evaluate_tracking(gt, frames({}, {}))
    assert m.mota == pytest.approx(0.0)  # 1 - 2/2
    assert m.idf1 == 0.0
    assert m.false_negatives == 2


def test_no_ground_truth_yields_zero_rather_than_dividing_by_zero():
    m = evaluate_tracking(frames({}), frames({"p1": BOX_A}))
    assert m.mota == 0.0
    assert m.idf1 == 0.0


# --- fragmentation ----------------------------------------------------------


def test_fragmentation_counts_resumption_after_a_gap():
    """Tracked, lost, tracked again -- one fragmentation, and notably zero ID
    switches because the same id was recovered. MOTA alone would not show it."""
    gt = frames({"g1": BOX_A}, {"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {}, {"p1": BOX_A})
    m = evaluate_tracking(gt, pred)
    assert m.fragmentations == 1
    assert m.id_switches == 0


def test_continuous_track_has_no_fragmentation():
    gt = frames({"g1": BOX_A}, {"g1": BOX_A})
    pred = frames({"p1": BOX_A}, {"p1": BOX_A})
    assert evaluate_tracking(gt, pred).fragmentations == 0


# --- matching tolerance -----------------------------------------------------


def test_slightly_offset_box_still_matches_above_threshold():
    gt = frames({"g1": BOX_A})
    pred = frames({"p1": BOX_A_SHIFTED})  # IoU ~0.82
    m = evaluate_tracking(gt, pred)
    assert m.true_positives == 1
    assert m.false_positives == 0


def test_raising_the_threshold_can_reject_a_loose_match():
    gt = frames({"g1": BOX_A})
    pred = frames({"p1": BOX_A_SHIFTED})
    m = evaluate_tracking(gt, pred, iou_threshold=0.9)
    assert m.true_positives == 0
    assert m.false_negatives == 1
