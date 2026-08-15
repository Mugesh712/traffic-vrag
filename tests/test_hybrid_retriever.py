"""Tests for M12's fusion ranking and Cypher templating — no live services."""
from __future__ import annotations

import pytest

from src.retrieval.hybrid_retriever import (
    build_count_query,
    build_event_query,
    build_object_query,
    build_vector_filter,
    fuse,
    rrf_score,
)
from src.retrieval.intent import parse_intent_rules
from src.utils.config import get_settings


def hit(global_id: str, rank: int, distance: float = 0.5, **metadata) -> dict:
    return {
        "global_id": global_id, "rank": rank, "distance": distance,
        "summary": f"summary for {global_id}", "metadata": {"class": "car", **metadata},
    }


def row(global_id: str, **props) -> dict:
    return {"global_id": global_id, "class": "car", **props}


# --- RRF -------------------------------------------------------------------


def test_rrf_normalized_so_rank_one_scores_one():
    """Raw RRF sits near 1/k (~0.016), which would make the weights
    meaningless against the other two [0,1] terms."""
    assert rrf_score(1, 60) == pytest.approx(1.0)


def test_rrf_decreases_monotonically_with_rank():
    scores = [rrf_score(r, 60) for r in range(1, 10)]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1.0 for s in scores)


# --- fusion ----------------------------------------------------------------


def test_object_found_by_both_retrievers_outranks_graph_only():
    settings = get_settings()
    intent = parse_intent_rules("white cars")
    fused = fuse(intent, [hit("obj_1", 1)], [row("obj_1"), row("obj_3")], settings)
    by_id = {o.global_id: o for o in fused}

    assert by_id["obj_1"].sources == ["vector", "graph"]
    assert by_id["obj_3"].sources == ["graph"]
    assert by_id["obj_1"].score > by_id["obj_3"].score


def test_unconstrained_question_unions_both_retrievers():
    settings = get_settings()
    intent = parse_intent_rules("show me something unusual")
    assert intent.has_structured_constraints() is False
    fused = fuse(intent, [hit("obj_1", 1), hit("obj_2", 2)], [row("obj_3")], settings)
    assert {o.global_id for o in fused} == {"obj_1", "obj_2", "obj_3"}


def test_hard_constraints_make_graph_membership_a_filter_not_a_bonus():
    """A vector hit that the KG did not return has failed an explicit
    constraint. Surfacing it anyway would answer "before 11:04" with an object
    first seen at 11:06 -- caught by querying the real stores."""
    settings = get_settings()
    intent = parse_intent_rules("cars before 11:04")
    assert intent.has_structured_constraints() is True
    fused = fuse(intent, [hit("in_range", 1), hit("out_of_range", 2)], [row("in_range")], settings)
    assert [o.global_id for o in fused] == ["in_range"]


def test_constraints_fall_back_to_union_when_the_graph_is_unreachable():
    """Degraded mode must still answer, since filtering on an absent graph
    would silently return nothing at all."""
    settings = get_settings()
    intent = parse_intent_rules("cars before 11:04")
    fused = fuse(intent, [hit("obj_1", 1)], [], settings, graph_available=False)
    assert [o.global_id for o in fused] == ["obj_1"]


def test_graph_only_object_still_surfaces():
    """Exact constraint satisfaction is authoritative -- an object the KG
    confirms must not be dropped just because the prose embedding missed it."""
    settings = get_settings()
    fused = fuse(parse_intent_rules("white cars"), [], [row("obj_9")], settings)
    assert [o.global_id for o in fused] == ["obj_9"]
    assert fused[0].graph_matched is True
    assert fused[0].score > 0


def test_confidence_term_breaks_ties_toward_the_more_certain_object():
    """The payoff from M6/M8: an object confirmed white at 1.0 outranks one
    where "white" won a coin flip the system flagged uncertain."""
    settings = get_settings()
    intent = parse_intent_rules("white cars")
    fused = fuse(
        intent,
        [hit("sure", 1), hit("unsure", 1)],  # identical vector rank
        [row("sure", color_confidence=1.0), row("unsure", color_confidence=0.2)],
        settings,
    )
    assert [o.global_id for o in fused] == ["sure", "unsure"]
    assert fused[0].matched_confidence == pytest.approx(1.0)


def test_confidence_term_is_neutral_when_no_attributes_were_asked_for():
    """Nothing was constrained, so there is nothing to be confident about; a
    blanket bonus would just add noise."""
    settings = get_settings()
    intent = parse_intent_rules("show me something unusual")
    fused = fuse(intent, [hit("obj_1", 1)], [row("obj_1", color_confidence=1.0)], settings)
    assert fused[0].matched_confidence == 0.0


def test_confidence_averages_across_every_constrained_attribute():
    settings = get_settings()
    intent = parse_intent_rules("white sedan")
    fused = fuse(
        intent, [hit("obj_1", 1)],
        [row("obj_1", color_confidence=1.0, vehicle_type_confidence=0.5)],
        settings,
    )
    assert fused[0].matched_confidence == pytest.approx(0.75)


def test_fusion_is_deterministic_for_tied_scores():
    settings = get_settings()
    intent = parse_intent_rules("show me something unusual")
    first = fuse(intent, [hit("b", 1), hit("a", 1)], [], settings)
    second = fuse(intent, [hit("a", 1), hit("b", 1)], [], settings)
    assert [o.global_id for o in first] == [o.global_id for o in second] == ["a", "b"]


def test_fusion_dedupes_by_global_id():
    settings = get_settings()
    fused = fuse(parse_intent_rules("cars"), [hit("obj_1", 1)], [row("obj_1")], settings)
    assert len(fused) == 1


# --- vector filter ---------------------------------------------------------


def test_vector_filter_uses_and_for_multiple_conditions():
    intent = parse_intent_rules("white cars that overtook")
    where = build_vector_filter("vid", intent)
    conditions = where["$and"]
    assert {"video_id": "vid"} in conditions
    assert {"color": "white"} in conditions
    assert {"event_OVERTAKE": True} in conditions


def test_vector_filter_is_flat_when_only_video_scoped():
    where = build_vector_filter("vid", parse_intent_rules("what happened?"))
    assert where == {"video_id": "vid"}


# --- Cypher templating -----------------------------------------------------


def test_object_query_binds_values_as_parameters_never_interpolated():
    """Injection safety: a hostile-looking value must reach Cypher only as a
    bound parameter, never spliced into the statement text."""
    intent = parse_intent_rules("white cars")
    intent.target_attributes["color"] = "white' OR 1=1 --"
    cypher, params = build_object_query("vid", intent, limit=10)
    assert "OR 1=1" not in cypher
    assert params["attr_color"] == "white' OR 1=1 --"
    assert "$attr_color" in cypher


def test_event_filter_uses_the_type_property_not_a_relationship_type():
    """Cypher cannot parameterize relationship types, so event filtering goes
    through Event.type -- which works because M10 kept full Event nodes."""
    cypher, params = build_object_query("vid", parse_intent_rules("cars that overtook"), limit=10)
    assert "type: $event_type_0" in cypher
    assert params["event_type_0"] == "OVERTAKE"
    assert "-[:OVERTAKES]->" not in cypher


def test_counterpart_constraint_traverses_the_shortcut_edge():
    """The relationship type is a code-defined literal (Cypher cannot
    parameterize types); the counterpart's properties stay bound parameters."""
    intent = parse_intent_rules("which white car overtook a blue truck?")
    cypher, params = build_object_query("vid", intent, limit=10)
    assert "-[:OVERTAKES]->(other:TrafficObject)" in cypher
    assert params["counterpart_classes"] == ["truck"]
    assert params["counterpart_color"] == "blue"
    # The subject's own filter stays white, not blue.
    assert params["attr_color"] == "white"


def test_event_predicate_is_subject_role_only():
    """"vehicles that overtook" must not also match the vehicle overtaken."""
    cypher, _ = build_object_query("vid", parse_intent_rules("cars that overtook"), limit=10)
    assert "role: 'subject'" in cypher


def test_time_bounds_compare_the_time_of_day_slice():
    intent = parse_intent_rules("cars before 12:05")
    cypher, params = build_object_query("vid", intent, limit=10)
    assert "substring(o.first_seen, 11, 8) <= $before" in cypher
    assert params["before"] == "12:05:00"


def test_unconstrained_query_has_no_where_clause():
    cypher, params = build_object_query("vid", parse_intent_rules("what happened?"), limit=10)
    assert "WHERE" not in cypher
    assert params == {"video_id": "vid", "limit": 10}


def test_count_query_groups_instead_of_returning_a_top_k_list():
    cypher, params = build_count_query("vid", parse_intent_rules("how many white cars?"))
    assert "count(*)" in cypher
    assert "LIMIT" not in cypher
    assert "limit" not in params
    assert params["attr_color"] == "white"


def test_count_query_keeps_the_same_predicates_as_the_object_query():
    intent = parse_intent_rules("how many white cars overtook before 12:05?")
    object_cypher, _ = build_object_query("vid", intent, limit=10)
    count_cypher, _ = build_count_query("vid", intent)
    for predicate in ("$attr_color", "$event_type_0", "$before"):
        assert predicate in object_cypher and predicate in count_cypher


def test_event_query_filters_by_type_and_time():
    intent = parse_intent_rules("what stopped after 11:30?")
    cypher, params = build_event_query("vid", intent, limit=10)
    assert "e.type IN $event_types" in cypher
    assert params["event_types"] == ["STOP"]
    assert params["after"] == "11:30:00"
