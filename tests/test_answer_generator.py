"""Tests for M13's pure parts: citation extraction/validation, context and
subgraph building, and the deterministic (no-LLM) paths. The Ollama backend
itself is exercised separately in test_answer_generator_integration.py.
"""
from __future__ import annotations

import pytest

from src.retrieval.answer_generator import (
    INSUFFICIENT_EVIDENCE,
    _evidence_frames_for,
    _format_count_answer,
    _resolve_timestamps,
    build_context,
    build_kg_subgraph,
    build_prompt,
    extract_citations,
    generate_answer,
    validate_citations,
)
from src.utils.config import get_settings
from src.utils.schemas import KGEdge, QueryIntent, RetrievalResult, RetrievedObject, TimestampSpan


def obj(global_id: str, cls="car", attributes=None, events=None, clips=None, frames=None,
        sighting_times=None) -> RetrievedObject:
    return RetrievedObject(
        global_id=global_id, **{"class": cls}, score=1.0,
        attributes=attributes or [], events=events or [],
        clips=clips or ["clip_000"], evidence_frames=frames or [],
        sighting_times=sighting_times or [],
        timeline_summary=f"summary for {global_id}",
    )


def attr(type_, value, confidence=1.0, winner=True, uncertain=False):
    return {"type": type_, "value": value, "confidence": confidence, "winner": winner, "uncertain": uncertain}


def event(event_id, type_, role="subject", start="2026-08-15T11:00:00", end="2026-08-15T11:00:30", confidence=0.9):
    return {"event_id": event_id, "type": type_, "role": role, "start_time": start, "end_time": end, "confidence": confidence}


def result(question="which car overtook?", objects=None, question_type="factual", count=None, breakdown=None):
    return RetrievalResult(
        question=question, video_id="v1",
        intent=QueryIntent(question=question, question_type=question_type),
        objects=objects or [], count=count, count_breakdown=breakdown or [],
    )


# --- citation extraction ----------------------------------------------------


def test_extract_citations_matches_the_required_format():
    text = "The white sedan [obj_0001 @ 11:03:20] overtook the truck [obj_0002 @ 11:03:20]."
    assert extract_citations(text) == [("obj_0001", "11:03:20"), ("obj_0002", "11:03:20")]


def test_extract_citations_returns_empty_for_uncited_text():
    assert extract_citations("The white sedan overtook the truck.") == []


def test_extract_citations_keeps_duplicates_in_order():
    text = "[obj_1 @ 11:00:00] parked, then [obj_1 @ 11:00:00] is mentioned again."
    assert extract_citations(text) == [("obj_1", "11:00:00"), ("obj_1", "11:00:00")]


# --- citation validation -----------------------------------------------------


def test_valid_citation_is_supported():
    times = {"obj_1": {"11:00:00"}}
    supported, unsupported = validate_citations([("obj_1", "11:00:00")], [obj("obj_1")], times)
    assert supported == ["obj_1"]
    assert unsupported == []


def test_citation_to_unretrieved_object_is_unsupported():
    """The core hallucination guard: a citation naming an object never in
    context must be flagged, not silently trusted."""
    supported, unsupported = validate_citations([("obj_99", "11:00:00")], [obj("obj_1")], {})
    assert supported == []
    assert unsupported == ["[obj_99 @ 11:00:00]"]


def test_citation_with_a_fabricated_timestamp_is_unsupported():
    """Observed live: a model named a real object but invented a timestamp
    ("00:00:00") that appeared nowhere in the context it was given. Checking
    only the object id would have wrongly accepted this."""
    times = {"obj_1": {"11:01:00"}}  # the only real timestamp obj_1 had
    supported, unsupported = validate_citations([("obj_1", "00:00:00")], [obj("obj_1")], times)
    assert supported == []
    assert unsupported == ["[obj_1 @ 00:00:00]"]


def test_object_with_no_events_has_no_citable_timestamp():
    """Conservative by construction: an object with nothing timestamped in
    context cannot be validly cited at all, rather than accepting any time."""
    supported, unsupported = validate_citations([("obj_1", "11:00:00")], [obj("obj_1")], {"obj_1": set()})
    assert supported == []
    assert unsupported == ["[obj_1 @ 11:00:00]"]


def test_validation_dedupes_repeated_citations_of_the_same_object():
    times = {"obj_1": {"11:00:00", "11:05:00"}}
    supported, _ = validate_citations(
        [("obj_1", "11:00:00"), ("obj_1", "11:05:00")], [obj("obj_1")], times
    )
    assert supported == ["obj_1"]  # object listed once, not per-citation


def test_validation_preserves_first_seen_order():
    times = {"obj_1": {"11:00:00"}, "obj_2": {"11:00:00"}}
    supported, _ = validate_citations(
        [("obj_2", "11:00:00"), ("obj_1", "11:00:00")], [obj("obj_1"), obj("obj_2")], times
    )
    assert supported == ["obj_2", "obj_1"]


# --- timestamp resolution -----------------------------------------------------


def test_citation_matching_an_event_start_resolves_to_the_event_span():
    o = obj("obj_1", events=[event("evt_1", "STOP", start="2026-08-15T11:03:20", end="2026-08-15T11:03:50")])
    spans = _resolve_timestamps([("obj_1", "11:03:20")], {"obj_1": o})
    assert spans == [TimestampSpan(start="11:03:20", end="11:03:50")]


def test_citation_with_no_matching_event_is_a_point_in_time():
    o = obj("obj_1")
    spans = _resolve_timestamps([("obj_1", "11:03:20")], {"obj_1": o})
    assert spans[0].start == spans[0].end == "11:03:20"


# --- context / prompt --------------------------------------------------------


def test_context_includes_only_confirmed_confident_attributes():
    o = obj("obj_1", attributes=[
        attr("color", "white"), attr("color", "silver", winner=False),
        attr("make", "toyota", uncertain=True),
    ])
    ctx, _ = build_context(result(objects=[o]), get_settings())
    assert "color=white" in ctx
    assert "silver" not in ctx
    assert "toyota" not in ctx  # uncertain, must not be presented as fact


def test_context_reports_exactly_the_timestamps_it_shows():
    """The set validate_citations checks against must match what a reader of
    the context block would actually see -- not a separately-derived guess."""
    o = obj("obj_1", events=[event("evt_1", "STOP", start="2026-08-15T11:00:00", end="2026-08-15T11:00:30")])
    ctx, valid_times = build_context(result(objects=[o]), get_settings())
    assert "11:00:00" in ctx and "11:00:30" in ctx
    assert valid_times["obj_1"] == {"11:00:00", "11:00:30"}


def test_context_gives_no_valid_times_for_an_object_with_no_events():
    o = obj("obj_1", attributes=[attr("color", "white")])
    _, valid_times = build_context(result(objects=[o]), get_settings())
    assert valid_times["obj_1"] == set()


def test_context_respects_max_context_objects(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.answer, "max_context_objects", 1)
    ctx, valid_times = build_context(result(objects=[obj("obj_1"), obj("obj_2")]), settings)
    assert "obj_1" in ctx
    assert "obj_2" not in ctx
    assert "obj_2" not in valid_times


def test_prompt_instructs_insufficient_evidence_and_gives_a_worked_citation_example():
    prompt = build_prompt("who stopped?", "- obj_1 (car):")
    assert INSUFFICIENT_EVIDENCE in prompt
    assert "who stopped?" in prompt
    # A worked example with a real-looking id/timestamp, not just an abstract
    # placeholder -- a weak model tends to echo a literal placeholder like
    # "global_id" back verbatim rather than substitute real values.
    assert "[obj_0007 @ 09:15:00]" in prompt
    assert 'never the literal words "global_id"' in prompt


# --- kg subgraph --------------------------------------------------------------


def test_subgraph_includes_only_supported_objects_not_the_full_retrieval():
    """The subgraph explains THIS answer, not everything M12 retrieved."""
    objects_by_id = {"obj_1": obj("obj_1"), "obj_2": obj("obj_2")}
    subgraph = build_kg_subgraph(["obj_1"], objects_by_id)
    ids = {n.id for n in subgraph.nodes}
    assert "obj_1" in ids
    assert "obj_2" not in ids


def test_subgraph_adds_event_nodes_and_involves_edges():
    o = obj("obj_1", events=[event("evt_1", "OVERTAKE", role="subject")])
    subgraph = build_kg_subgraph(["obj_1"], {"obj_1": o})
    labels = {n.id: n.label for n in subgraph.nodes}
    assert labels["evt_1"] == "Event"
    assert labels["obj_1"] == "TrafficObject"
    assert subgraph.edges == [
        KGEdge(source="evt_1", target="obj_1", type="INVOLVES", properties={"role": "subject"})
    ]


def test_subgraph_deduplicates_an_event_shared_by_two_objects():
    shared_event = event("evt_1", "OVERTAKE", role="subject")
    other_role_event = event("evt_1", "OVERTAKE", role="object")
    objects_by_id = {
        "obj_1": obj("obj_1", events=[shared_event]),
        "obj_2": obj("obj_2", events=[other_role_event]),
    }
    subgraph = build_kg_subgraph(["obj_1", "obj_2"], objects_by_id)
    event_nodes = [n for n in subgraph.nodes if n.label == "Event"]
    assert len(event_nodes) == 1  # one Event node, not one per participant
    assert len(subgraph.edges) == 2  # but both INVOLVES edges present


# --- evidence frames grouped by object -----------------------------------------
# Grouped rather than pooled into one flat list, so the UI can show "this is
# obj_0004" next to obj_0004's own citation instead of an undifferentiated
# photo strip the reader has to match up by hand.


def test_evidence_frames_are_grouped_by_the_object_they_show():
    objects_by_id = {
        "obj_1": obj("obj_1", frames=["f1.jpg", "f2.jpg"]),
        "obj_2": obj("obj_2", frames=["f3.jpg"]),
    }
    frames_by_object = _evidence_frames_for(["obj_1", "obj_2"], objects_by_id)
    assert frames_by_object == {"obj_1": ["f1.jpg", "f2.jpg"], "obj_2": ["f3.jpg"]}


def test_evidence_frames_include_only_supported_objects():
    objects_by_id = {"obj_1": obj("obj_1", frames=["f1.jpg"]), "obj_2": obj("obj_2", frames=["f2.jpg"])}
    frames_by_object = _evidence_frames_for(["obj_1"], objects_by_id)
    assert "obj_2" not in frames_by_object


def test_object_with_no_evidence_frames_is_absent_not_an_empty_list():
    objects_by_id = {"obj_1": obj("obj_1", frames=[])}
    frames_by_object = _evidence_frames_for(["obj_1"], objects_by_id)
    assert frames_by_object == {}


# --- deterministic paths: counting and empty results --------------------------


def test_counting_question_never_calls_the_llm(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("LLM must not be called for a counting question")
    monkeypatch.setattr("src.retrieval.answer_generator._load_backend", fail)

    r = result(question_type="counting", count=3, breakdown=[{"n": 3, "class": "car", "color": "white"}])
    answer = generate_answer(r, settings=get_settings())
    assert answer.status == "counting"
    assert "3" in answer.answer
    assert "LLM was not called" in answer.reasoning_trace


def test_empty_retrieval_never_calls_the_llm(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("LLM must not be called when nothing was retrieved")
    monkeypatch.setattr("src.retrieval.answer_generator._load_backend", fail)

    answer = generate_answer(result(objects=[]), settings=get_settings())
    assert answer.status == "insufficient_evidence"
    assert answer.answer == INSUFFICIENT_EVIDENCE


def test_format_count_answer_describes_the_breakdown():
    r = result(question_type="counting", count=2, breakdown=[
        {"n": 1, "class": "car", "color": "white", "vehicle_type": "sedan"},
        {"n": 1, "class": "truck", "color": "blue", "vehicle_type": None},
    ])
    text = _format_count_answer(r)
    assert "2" in text
    assert "white" in text and "blue" in text


# --- citable sighting times for event-less objects --------------------------
#
# Citations carry a timestamp, and only cited objects become evidence frames or
# subgraph nodes. An object with no events showed no timestamp in its context
# block, so every citation of it was rejected and answers about it grounded
# nothing -- observed live as an answer with empty evidence and empty subgraph.


def test_event_less_object_still_offers_a_citable_time():
    o = obj("obj_1", events=[], sighting_times=["2026-08-15T11:02:03.456789"])

    ctx, valid_times = build_context(result(objects=[o]), get_settings())

    assert "[obj_1 @ 11:02:03]" in ctx
    assert "11:02:03" in valid_times["obj_1"]


def test_citation_of_an_event_less_object_now_validates():
    o = obj("obj_1", events=[], sighting_times=["2026-08-15T11:02:03"])
    _, valid_times = build_context(result(objects=[o]), get_settings())

    supported, unsupported = validate_citations([("obj_1", "11:02:03")], [o], valid_times)

    assert supported == ["obj_1"]
    assert unsupported == []


def test_earliest_sighting_is_the_one_offered():
    """One time is offered per object, so which one has to be defined rather
    than whichever row the graph happened to return first."""
    o = obj("obj_1", events=[], sighting_times=["2026-08-15T11:00:05", "2026-08-15T11:09:00"])

    ctx, valid_times = build_context(result(objects=[o]), get_settings())

    assert "[obj_1 @ 11:00:05]" in ctx
    assert "11:09:00" not in valid_times["obj_1"]


def test_sighting_time_does_not_weaken_fabricated_timestamp_rejection():
    """This widens what can be cited; it must not widen what passes validation."""
    o = obj("obj_1", events=[], sighting_times=["2026-08-15T11:02:03"])
    _, valid_times = build_context(result(objects=[o]), get_settings())

    supported, unsupported = validate_citations([("obj_1", "00:00:00")], [o], valid_times)

    assert supported == []
    assert unsupported == ["[obj_1 @ 00:00:00]"]


def test_event_times_and_sighting_time_are_both_citable():
    o = obj(
        "obj_1",
        events=[event("evt_1", "STOP", start="2026-08-15T11:05:00", end="2026-08-15T11:05:30")],
        sighting_times=["2026-08-15T11:02:03"],
    )

    _, valid_times = build_context(result(objects=[o]), get_settings())

    assert valid_times["obj_1"] == {"11:02:03", "11:05:00", "11:05:30"}


def test_prompt_restates_the_citation_format_next_to_the_question():
    """The rule appears in the preamble, but a small model follows the last
    instruction it read. Measured: with the rule only at the top, qwen2.5:3b
    answered "obj_0004 (car) overtook obj_0009 (car)" -- real objects in a
    shape the citation regex cannot match, leaving the answer uncited and its
    evidence and subgraph empty."""
    prompt = build_prompt("who overtook?", "- obj_1 (car):")

    reminder_at = prompt.rfind("obj_9999")
    assert reminder_at != -1
    # Asks for a sentence too: demanding the citation alone made the model drop
    # the prose and answer "[obj_0004 @ 18:15:17] to [obj_0004 @ 18:15:19]",
    # which says when but not what.
    assert "normal sentence" in prompt
    # After the context, so it is the last formatting instruction before the
    # model starts writing.
    assert reminder_at > prompt.find("CONTEXT:")
    assert reminder_at < prompt.find("ANSWER:")


def test_reminder_example_uses_an_id_no_real_context_can_contain():
    """The reminder needs a concrete example to imitate, but an example built
    from real ids is indistinguishable from an answer: with one, the model
    replayed it verbatim for an unrelated question, and because those ids were
    in context the copy partly validated -- a wrong answer carrying supported
    citations. A sentinel id keeps the example imitable while guaranteeing a
    verbatim echo is rejected as unsupported instead."""
    prompt = build_prompt("who stopped?", "- obj_0004 (car):")

    assert "obj_9999" in prompt
    o = obj("obj_0004", events=[], sighting_times=["2026-08-15T11:02:03"])
    _, valid_times = build_context(result(objects=[o]), get_settings())
    supported, unsupported = validate_citations(
        extract_citations("The blue van [obj_9999 @ 09:15:00] stopped at the junction."),
        [o],
        valid_times,
    )
    assert supported == []
    assert unsupported == ["[obj_9999 @ 09:15:00]"]
