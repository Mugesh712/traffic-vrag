"""Tests for M16's attribute metrics and ground-truth handling."""
from __future__ import annotations

import json

import pytest

from src.eval.attributes import (
    evaluate_attribute_accuracy,
    evaluate_attribute_consistency,
    summarize_corrections,
    uncertainty_rate,
)
from src.eval.ground_truth import (
    UNREVIEWED_MARKER,
    GroundTruth,
    GroundTruthError,
    load_ground_truth,
)
from src.utils.config import get_settings
from src.utils.schemas import (
    CanonicalAttributes,
    ClipCanonicalAttributes,
    CorrectionLogEntry,
    FinalAttribute,
    FinalObjectIndex,
    GlobalObjectFinal,
    Sighting,
    VoteDistribution,
)


def final_object(global_id, attrs, sightings=None):
    return GlobalObjectFinal(
        global_id=global_id,
        **{"class": "car"},
        sightings=sightings or [Sighting(clip_id="c1", track_id="1", first_seen="x", last_seen="y")],
        attributes=[
            FinalAttribute(attribute=k, value=v[0], confidence=1.0, source="agreed", uncertain=v[1])
            for k, v in attrs.items()
        ],
    )


def vote(attribute, winner, uncertain=False):
    return VoteDistribution(
        attribute=attribute, winner=winner, distribution={winner: 1.0} if winner else {},
        uncertain=uncertain, n_votes=3, margin=1.0,
    )


# --- accuracy ---------------------------------------------------------------


def test_correct_value_scores_correct():
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {"color": ("white", False)})])
    gt = GroundTruth(video_id="v", attributes=[{"gt_id": "gt_1", "color": "white"}])
    results = {r.attribute: r for r in evaluate_attribute_accuracy(final, gt, {"obj_1": "gt_1"})}
    assert results["color"].correct == 1
    assert results["color"].wrong == 0


def test_wrong_value_scores_wrong():
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {"color": ("blue", False)})])
    gt = GroundTruth(video_id="v", attributes=[{"gt_id": "gt_1", "color": "white"}])
    results = {r.attribute: r for r in evaluate_attribute_accuracy(final, gt, {"obj_1": "gt_1"})}
    assert results["color"].wrong == 1
    assert results["color"].correct == 0


def test_uncertain_prediction_is_an_abstention_not_an_error():
    """The distinction the whole project rests on: declining to answer is not
    the same failure as answering wrongly."""
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {"color": ("blue", True)})])
    gt = GroundTruth(video_id="v", attributes=[{"gt_id": "gt_1", "color": "white"}])
    results = {r.attribute: r for r in evaluate_attribute_accuracy(final, gt, {"obj_1": "gt_1"})}
    assert results["color"].abstained == 1
    assert results["color"].wrong == 0
    assert results["color"].correct == 0


def test_precision_ignores_abstentions_but_coverage_reports_them():
    final = FinalObjectIndex(
        video_id="v",
        objects=[
            final_object("obj_1", {"color": ("white", False)}),
            final_object("obj_2", {"color": ("blue", False)}),
            final_object("obj_3", {"color": ("red", True)}),
        ],
    )
    gt = GroundTruth(
        video_id="v",
        attributes=[
            {"gt_id": "gt_1", "color": "white"},
            {"gt_id": "gt_2", "color": "white"},
            {"gt_id": "gt_3", "color": "white"},
        ],
    )
    mapping = {"obj_1": "gt_1", "obj_2": "gt_2", "obj_3": "gt_3"}
    results = {r.attribute: r for r in evaluate_attribute_accuracy(final, gt, mapping)}
    color = results["color"]
    assert color.precision_when_answering == pytest.approx(0.5)  # 1 of 2 answered
    assert color.coverage == pytest.approx(2 / 3)  # answered 2 of 3


def test_object_with_no_annotation_is_not_scored():
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {"color": ("white", False)})])
    gt = GroundTruth(video_id="v", attributes=[])
    results = {r.attribute: r for r in evaluate_attribute_accuracy(final, gt, {})}
    assert results["color"].missing_gt == 1
    assert results["color"].answered == 0


def test_direction_is_excluded_from_identity_scoring():
    """State, not identity -- consistent with M7/M8's own exclusion."""
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {"color": ("white", False)})])
    gt = GroundTruth(video_id="v", attributes=[{"gt_id": "gt_1", "color": "white"}])
    scored = {r.attribute for r in evaluate_attribute_accuracy(final, gt, {"obj_1": "gt_1"})}
    assert "direction" not in scored
    assert "color" in scored


# --- consistency ------------------------------------------------------------


def two_sightings():
    return [
        Sighting(clip_id="c1", track_id="1", first_seen="x", last_seen="y"),
        Sighting(clip_id="c2", track_id="2", first_seen="x", last_seen="y"),
    ]


def canonical_for(pairs):
    """pairs: {(clip_id, track_id): [VoteDistribution, ...]}"""
    by_clip = {}
    for (clip_id, track_id), votes in pairs.items():
        by_clip.setdefault(clip_id, []).append(CanonicalAttributes(track_id=track_id, votes=votes))
    return {c: ClipCanonicalAttributes(clip_id=c, tracks=t) for c, t in by_clip.items()}


def test_object_agreeing_across_clips_is_consistent():
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {}, two_sightings())])
    canonical = canonical_for({("c1", "1"): [vote("color", "white")], ("c2", "2"): [vote("color", "white")]})
    results = {r.attribute: r for r in evaluate_attribute_consistency(final, canonical)}
    assert results["color"].consistent_objects == 1
    assert results["color"].inconsistent_objects == 0


def test_object_disagreeing_across_clips_is_inconsistent():
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {}, two_sightings())])
    canonical = canonical_for({("c1", "1"): [vote("color", "white")], ("c2", "2"): [vote("color", "blue")]})
    results = {r.attribute: r for r in evaluate_attribute_consistency(final, canonical)}
    assert results["color"].inconsistent_objects == 1


def test_uncertain_sighting_abstains_rather_than_disagreeing():
    """Same permissive treatment M7's semantic gate gives uncertainty."""
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {}, two_sightings())])
    canonical = canonical_for({
        ("c1", "1"): [vote("color", "white")],
        ("c2", "2"): [vote("color", "blue", uncertain=True)],
    })
    results = {r.attribute: r for r in evaluate_attribute_consistency(final, canonical)}
    assert results["color"].inconsistent_objects == 0
    assert results["color"].single_sighting_objects == 1  # only one confident value left


def test_single_clip_object_is_excluded_from_the_rate():
    """Cannot disagree with itself; counting it as consistent would inflate
    the rate to a meaningless 100% on single-clip footage."""
    final = FinalObjectIndex(video_id="v", objects=[final_object("obj_1", {})])
    canonical = canonical_for({("c1", "1"): [vote("color", "white")]})
    results = {r.attribute: r for r in evaluate_attribute_consistency(final, canonical)}
    assert results["color"].single_sighting_objects == 1
    assert results["color"].consistent_objects == 0
    assert results["color"].rate == 0.0  # undefined, reported as 0 with the exclusion count


# --- correction summary / uncertainty ---------------------------------------


def test_correction_summary_counts_each_outcome():
    final = FinalObjectIndex(
        video_id="v",
        correction_log=[
            CorrectionLogEntry(global_id="o1", attribute="color", outcome="confirmed"),
            CorrectionLogEntry(global_id="o1", attribute="make", outcome="filled"),
            CorrectionLogEntry(global_id="o2", attribute="make", outcome="corrected"),
            CorrectionLogEntry(global_id="o2", attribute="color", outcome="retained"),
        ],
    )
    summary = summarize_corrections(final)
    assert (summary.confirmed, summary.filled, summary.corrected, summary.retained) == (1, 1, 1, 1)
    assert summary.per_attribute["make"] == {"filled": 1, "corrected": 1}


def test_uncertainty_rate_per_attribute():
    canonical = canonical_for({
        ("c1", "1"): [vote("color", "white"), vote("make", None, uncertain=True)],
        ("c1", "2"): [vote("color", "blue", uncertain=True), vote("make", None, uncertain=True)],
    })
    rates = uncertainty_rate(canonical)
    assert rates["color"] == pytest.approx(0.5)
    assert rates["make"] == pytest.approx(1.0)


# --- ground truth loading ---------------------------------------------------


def test_missing_ground_truth_returns_none_rather_than_raising():
    """The harness must still run and report its GT-free metrics."""
    assert load_ground_truth("no_such_video_at_all", get_settings()) is None


def test_unreviewed_template_is_refused(tmp_path, monkeypatch):
    settings = get_settings()
    gt_dir = settings.resolve_path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / "video_test_m16_unreviewed.json"
    path.write_text(json.dumps({
        UNREVIEWED_MARKER: "not reviewed",
        "video_id": "video_test_m16_unreviewed", "tracks": [], "attributes": [],
    }))
    try:
        with pytest.raises(GroundTruthError, match="unreviewed template"):
            load_ground_truth("video_test_m16_unreviewed", settings)
    finally:
        path.unlink(missing_ok=True)


def test_reviewed_ground_truth_loads():
    settings = get_settings()
    gt_dir = settings.resolve_path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / "video_test_m16_reviewed.json"
    path.write_text(json.dumps({
        "video_id": "video_test_m16_reviewed",
        "tracks": [{"gt_id": "gt_1", "class": "car", "boxes": {"f0": [0, 0, 10, 10]}}],
        "attributes": [{"gt_id": "gt_1", "color": "white"}],
    }))
    try:
        gt = load_ground_truth("video_test_m16_reviewed", settings)
        assert gt is not None
        assert gt.boxes_for_frame("f0") == {"gt_1": [0.0, 0.0, 10.0, 10.0]}
        assert gt.attributes_by_id()["gt_1"].color == "white"
    finally:
        path.unlink(missing_ok=True)
