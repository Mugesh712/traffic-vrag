"""Tests for M11's pure summary/metadata builders — no ChromaDB needed."""
from __future__ import annotations

from src.retrieval.vector_store import (
    EVENT_TYPES,
    _attr,
    _descriptor,
    _first_last_seen,
    _format_time,
    build_event_metadata,
    build_event_summary,
    build_object_metadata,
    build_object_summary,
)
import pytest

from src.utils.config import get_settings
from src.utils.schemas import Event, FinalAttribute, GlobalObjectFinal, Sighting


def obj(global_id: str, cls: str = "car", attrs: dict | None = None, sightings=None) -> GlobalObjectFinal:
    attrs = attrs or {}
    attributes = [
        FinalAttribute(
            attribute=name,
            value=val.get("value") if isinstance(val, dict) else val,
            confidence=val.get("confidence", 1.0) if isinstance(val, dict) else 1.0,
            source=val.get("source", "agreed") if isinstance(val, dict) else "agreed",
            uncertain=val.get("uncertain", False) if isinstance(val, dict) else False,
        )
        for name, val in attrs.items()
    ]
    return GlobalObjectFinal(
        global_id=global_id,
        **{"class": cls},
        sightings=sightings or [Sighting(clip_id="c1", track_id="1",
                                          first_seen="2026-08-15T11:02:10", last_seen="2026-08-15T11:02:45")],
        attributes=attributes,
        best_shot_crops=[],
    )


def event(event_id, etype, subject, obj_id=None, start="2026-08-15T11:04:20",
          end="2026-08-15T11:04:25", metadata=None) -> Event:
    return Event(
        event_id=event_id, type=etype, subject_id=subject, object_id=obj_id,
        start_time=start, end_time=end, confidence=0.9,
        evidence_frames=[], metadata=metadata or {},
    )


# --- pure helpers -----------------------------------------------------------


def test_format_time_drops_date_and_fraction():
    assert _format_time("2026-08-15T11:04:20.04") == "11:04:20"
    assert _format_time("2026-08-15T11:04:20") == "11:04:20"


def test_attr_ignores_uncertain_values():
    o = obj("obj_1", attrs={"color": {"value": "silver", "uncertain": True}})
    assert _attr(o, "color") is None


def test_attr_returns_confirmed_value():
    o = obj("obj_1", attrs={"color": "white"})
    assert _attr(o, "color") == "white"


def test_first_last_seen_spans_all_sightings():
    o = obj("obj_1", sightings=[
        Sighting(clip_id="c1", track_id="1", first_seen="2026-08-15T11:00:00", last_seen="2026-08-15T11:00:30"),
        Sighting(clip_id="c2", track_id="2", first_seen="2026-08-15T11:01:00", last_seen="2026-08-15T11:01:30"),
    ])
    first, last = _first_last_seen(o)
    assert first == "2026-08-15T11:00:00"
    assert last == "2026-08-15T11:01:30"


# --- descriptor ---------------------------------------------------------


def test_descriptor_uses_all_confirmed_attributes():
    o = obj("obj_1", attrs={"color": "white", "make": "toyota", "vehicle_type": "sedan"})
    assert _descriptor(o) == "White Toyota sedan"


def test_descriptor_falls_back_to_class_when_nothing_confirmed():
    o = obj("obj_1", cls="truck")
    assert _descriptor(o) == "truck"


def test_descriptor_skips_uncertain_attributes_individually():
    o = obj("obj_1", attrs={"color": "white", "make": {"value": "toyota", "uncertain": True}})
    assert _descriptor(o) == "White car"  # make dropped, class fallback for type


# --- object summary ----------------------------------------------------


def test_object_summary_matches_roadmap_shape():
    """"White Toyota sedan, first seen 11:02:10 heading east, overtook a blue
    truck at 11:04:20, stopped at 11:05:42." -- the exact worked example from
    the roadmap's M11 spec."""
    subject = obj("obj_1", attrs={"color": "white", "make": "toyota", "vehicle_type": "sedan", "direction": "east"})
    other = obj("obj_2", cls="truck", attrs={"color": "blue"})
    events = [
        event("evt_1", "OVERTAKE", "obj_1", "obj_2", start="2026-08-15T11:04:20"),
        event("evt_2", "STOP", "obj_1", start="2026-08-15T11:05:42"),
    ]
    summary = build_object_summary(subject, events, {"obj_1": subject, "obj_2": other})
    assert summary == (
        "White Toyota sedan, first seen 11:02:10 heading east, "
        "overtook a blue truck at 11:04:20, stopped at 11:05:42."
    )


def test_object_summary_without_events_shows_the_observed_span():
    o = obj("obj_1", attrs={"color": "white"}, sightings=[
        Sighting(clip_id="c1", track_id="1", first_seen="2026-08-15T11:02:10", last_seen="2026-08-15T11:02:45"),
    ])
    summary = build_object_summary(o, [], {"obj_1": o})
    assert summary == "White car, seen from 11:02:10 to 11:02:45."


def test_object_summary_uses_article_correctly_for_vowel_start():
    subject = obj("obj_1", attrs={"color": "white"})
    other = obj("obj_2", attrs={"vehicle_type": "ambulance"})
    events = [event("evt_1", "OVERTAKE", "obj_1", "obj_2")]
    summary = build_object_summary(subject, events, {"obj_1": subject, "obj_2": other})
    assert "overtook an ambulance" in summary


def test_object_summary_describes_being_overtaken_from_the_other_side():
    """The object on the receiving end of an OVERTAKE gets its own clause,
    phrased passively, not the overtaker's active one."""
    overtaker = obj("obj_1", attrs={"color": "white"})
    subject = obj("obj_2", cls="truck", attrs={"color": "blue"})
    events = [event("evt_1", "OVERTAKE", "obj_1", "obj_2")]
    summary = build_object_summary(subject, events, {"obj_1": overtaker, "obj_2": subject})
    assert "was overtaken by a white car" in summary


def test_object_summary_orders_events_chronologically_regardless_of_input_order():
    o = obj("obj_1", attrs={"color": "white"})
    events = [
        event("evt_2", "PARK", "obj_1", start="2026-08-15T11:10:00"),
        event("evt_1", "STOP", "obj_1", start="2026-08-15T11:05:00"),
    ]
    summary = build_object_summary(o, events, {"obj_1": o})
    assert summary.index("stopped") < summary.index("parked")


def test_turn_and_crosses_phrases_use_metadata():
    o = obj("obj_1", attrs={"color": "white"})
    events = [
        event("evt_1", "TURN", "obj_1", metadata={"turn_direction": "left"}, start="2026-08-15T11:00:00"),
        event("evt_2", "CROSSES", "obj_1", metadata={"region_id": "stop_line"}, start="2026-08-15T11:01:00"),
    ]
    summary = build_object_summary(o, events, {"obj_1": o})
    assert "turned left" in summary
    assert "crossed stop_line" in summary


# --- object metadata -----------------------------------------------------


def test_object_metadata_sets_boolean_flags_for_involved_event_types_only():
    o = obj("obj_1", attrs={"color": "white"})
    events = [event("evt_1", "OVERTAKE", "obj_1"), event("evt_2", "STOP", "obj_2")]  # obj_2 not involved
    meta = build_object_metadata("video_1", o, events)
    assert meta["event_OVERTAKE"] is True
    assert meta["event_STOP"] is False
    assert set(f"event_{t}" for t in EVENT_TYPES) <= meta.keys()


def test_object_metadata_event_flags_are_subject_role_only():
    """event_OVERTAKE=true must mean "did the overtaking", not "was involved
    in any role" -- the vehicle that got overtaken must not match a filter
    for vehicles that overtook something."""
    overtaker = obj("obj_1")
    victim = obj("obj_2")
    events = [event("evt_1", "OVERTAKE", "obj_1", "obj_2")]
    overtaker_meta = build_object_metadata("video_1", overtaker, events)
    victim_meta = build_object_metadata("video_1", victim, events)
    assert overtaker_meta["event_OVERTAKE"] is True
    assert victim_meta["event_OVERTAKE"] is False
    # But the victim's overall event count still reflects its involvement.
    assert victim_meta["n_events"] == 1


def test_object_metadata_supports_the_roadmap_filter_pattern():
    """color=white AND event_type=OVERTAKE, as two boolean/string equalities."""
    o = obj("obj_1", attrs={"color": "white"})
    meta = build_object_metadata("video_1", o, [event("evt_1", "OVERTAKE", "obj_1")])
    assert meta["color"] == "white"
    assert meta["event_OVERTAKE"] is True


def test_object_metadata_uses_unknown_not_none_for_missing_attributes():
    """Chroma metadata values must be scalars; None would be invalid."""
    o = obj("obj_1")
    meta = build_object_metadata("video_1", o, [])
    assert meta["color"] == "unknown"
    assert meta["make"] == "unknown"
    assert all(v is not None for v in meta.values())


def test_object_metadata_counts_only_events_this_object_participated_in():
    o = obj("obj_1")
    events = [
        event("evt_1", "OVERTAKE", "obj_1", "obj_9"),
        event("evt_2", "STOP", "obj_9"),  # unrelated
    ]
    meta = build_object_metadata("video_1", o, events)
    assert meta["n_events"] == 1


# --- event summary / metadata --------------------------------------------


def test_event_summary_names_both_participants_in_an_overtake():
    subject = obj("obj_1", attrs={"color": "white", "vehicle_type": "sedan"})
    other = obj("obj_2", cls="truck", attrs={"color": "blue"})
    e = event("evt_1", "OVERTAKE", "obj_1", "obj_2", start="2026-08-15T11:04:20")
    assert build_event_summary(e, {"obj_1": subject, "obj_2": other}) == (
        "White sedan overtook a blue truck at 11:04:20."
    )


def test_event_summary_handles_unknown_subject_gracefully():
    e = event("evt_1", "STOP", "obj_missing", start="2026-08-15T11:00:00")
    assert build_event_summary(e, {}) == "A vehicle stopped at 11:00:00."


def test_event_metadata_defaults_object_id_to_empty_string_not_none():
    e = event("evt_1", "STOP", "obj_1")
    meta = build_event_metadata("video_1", e, {"obj_1": obj("obj_1")})
    assert meta["object_id"] == ""
    assert all(v is not None for v in meta.values())


# --- client mode selection (M17) --------------------------------------------


def test_embedded_is_the_default_mode():
    """The CLI on a laptop must work with no server running."""
    assert get_settings().vector_store.mode == "embedded"


def test_embedded_mode_returns_a_persistent_client(monkeypatch, tmp_path):
    from src.retrieval.vector_store import get_chroma_client

    settings = get_settings()
    monkeypatch.setattr(settings.vector_store, "mode", "embedded")
    monkeypatch.setattr(settings.vector_store, "persist_dir", str(tmp_path / "chroma"))
    client = get_chroma_client(settings)
    assert client.__class__.__name__ in ("Client", "PersistentClient", "ClientCreator")


def test_unknown_mode_is_a_clear_error(monkeypatch):
    """A typo in the mode must fail loudly, not fall through to a default."""
    from src.retrieval.vector_store import VectorStoreError, get_chroma_client

    settings = get_settings()
    monkeypatch.setattr(settings.vector_store, "mode", "sqlite")
    with pytest.raises(VectorStoreError, match="Unknown vector_store.mode"):
        get_chroma_client(settings)
