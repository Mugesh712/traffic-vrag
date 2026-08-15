"""Tests for M10's payload builder — the graph's contents, without a database.

Fixtures are written under the real data/ tree and cleaned up unconditionally.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graph.kg_builder import (
    KGBuilderError,
    _aggregate_distributions,
    _schema_statements,
    _uid,
    collect_payload,
    split_cypher_statements,
)
from src.utils.config import get_settings

VIDEO_ID = "video_test_m10"
CLIP_A, CLIP_B = "clip_test_m10_a", "clip_test_m10_b"


def frame_row(clip_id: str, frame_id: str, ts: float) -> dict:
    return {
        "clip_id": clip_id, "frame_id": frame_id,
        "frame_path": f"data/frames/{clip_id}/{frame_id}.jpg",
        "video_timestamp_sec": ts, "wallclock_time": f"2026-08-15T10:00:{ts:05.2f}",
        "fps": 25.0, "resolution": [640, 480],
    }


def clip_row(clip_id: str, frames: list[dict]) -> dict:
    return {
        "video_id": VIDEO_ID, "clip_id": clip_id, "clip_path": f"{clip_id}.mp4",
        "start_wallclock": "2026-08-15T10:00:00", "end_wallclock": "2026-08-15T10:00:04",
        "fps": 25.0, "resolution": [640, 480], "frames": frames,
    }


def track_row(track_id: str, frames: list[str]) -> dict:
    return {
        "track_id": track_id, "class": "car", "frames": frames,
        "bboxes": [[10.0, 20.0, 60.0, 70.0]] * len(frames),
        "centers": [[35.0, 45.0]] * len(frames),
        "velocity": [[0.0, 0.0]] * len(frames),
        "dominant_direction_deg": None, "embedding": [],
        "best_shot_crops": [], "best_shot_scores": [],
    }


def vote_row(attribute: str, winner, distribution: dict, uncertain=False, n_votes=8) -> dict:
    return {
        "attribute": attribute, "winner": winner, "distribution": distribution,
        "uncertain": uncertain, "n_votes": n_votes, "margin": 1.0, "uncertain_reason": None,
    }


@pytest.fixture
def fixture(tmp_path_factory):
    settings = get_settings()
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    written: list = []

    def write(sub: str, name: str, data: dict):
        path = outputs / sub / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        written.append(path)

    frames_a = [frame_row(CLIP_A, "f0", 0.0), frame_row(CLIP_A, "f1", 1.0)]
    frames_b = [frame_row(CLIP_B, "f2", 2.0), frame_row(CLIP_B, "f3", 3.0)]

    write("ingest", f"{VIDEO_ID}_manifest.json", {
        "video_id": VIDEO_ID, "source_path": "fixture.mp4", "fps": 25.0,
        "resolution": [640, 480], "frame_count": 4,
        "start_wallclock": "2026-08-15T10:00:00", "end_wallclock": "2026-08-15T10:00:04",
        "clips": [clip_row(CLIP_A, frames_a), clip_row(CLIP_B, frames_b)],
    })

    write("tracks_associated", f"{CLIP_A}.json", {
        "clip_id": CLIP_A, "tracks": [track_row("1", ["f0", "f1"])],
        "merge_log": [], "rejected_links": [], "suppressed_links": [],
    })
    write("tracks_associated", f"{CLIP_B}.json", {
        "clip_id": CLIP_B, "tracks": [track_row("7", ["f2", "f3"])],
        "merge_log": [], "rejected_links": [], "suppressed_links": [],
    })

    # Clip A saw it mostly white; clip B was less sure.
    write("attributes_canonical", f"{CLIP_A}.json", {
        "clip_id": CLIP_A, "tracks": [{"track_id": "1", "votes": [
            vote_row("color", "white", {"white": 0.8, "silver": 0.2}, n_votes=10),
            vote_row("vehicle_type", "sedan", {"sedan": 1.0}, n_votes=10),
        ]}],
    })
    write("attributes_canonical", f"{CLIP_B}.json", {
        "clip_id": CLIP_B, "tracks": [{"track_id": "7", "votes": [
            vote_row("color", "silver", {"silver": 0.6, "white": 0.4}, uncertain=True, n_votes=5),
        ]}],
    })

    write("global_objects_final", f"{VIDEO_ID}.json", {
        "video_id": VIDEO_ID,
        "objects": [{
            "global_id": "obj_0001", "class": "car",
            "sightings": [
                {"clip_id": CLIP_A, "track_id": "1", "first_seen": "2026-08-15T10:00:00",
                 "last_seen": "2026-08-15T10:00:01", "gate_scores": {}},
                {"clip_id": CLIP_B, "track_id": "7", "first_seen": "2026-08-15T10:00:02",
                 "last_seen": "2026-08-15T10:00:03", "gate_scores": {}},
            ],
            "attributes": [
                {"attribute": "color", "value": "white", "confidence": 0.9,
                 "source": "agreed", "uncertain": False},
                {"attribute": "make", "value": "toyota", "confidence": 1.0,
                 "source": "best_shot", "uncertain": False},
            ],
            "best_shot_crops": [],
        }],
        "correction_log": [],
    })

    write("events", f"{VIDEO_ID}.json", {"events": [
        {"event_id": "evt_00001", "type": "STOP", "subject_id": "obj_0001",
         "object_id": None, "start_time": "2026-08-15T10:00:01",
         "end_time": "2026-08-15T10:00:03", "confidence": 0.9,
         "evidence_frames": ["f1", "f2"], "metadata": {}},
    ]})

    yield settings

    for path in written:
        path.unlink(missing_ok=True)


# --- identifier scoping ----------------------------------------------------


def test_uid_is_video_scoped():
    """obj_0001 recurs in every video, so raw ids cannot be unique keys."""
    assert _uid("vid_a", "obj_0001") != _uid("vid_b", "obj_0001")


# --- distribution aggregation ----------------------------------------------


def test_distributions_weighted_by_evidence_across_sightings():
    canonical = {
        "c1": {"1": {"color": {"distribution": {"white": 1.0}, "n_votes": 20}}},
        "c2": {"2": {"color": {"distribution": {"silver": 1.0}, "n_votes": 5}}},
    }
    result = _aggregate_distributions(canonical, [("c1", "1"), ("c2", "2")])
    assert result["color"]["white"] == pytest.approx(20 / 25)
    assert result["color"]["silver"] == pytest.approx(5 / 25)


def test_distribution_absent_when_no_votes():
    assert _aggregate_distributions({}, [("c1", "1")]) == {}


# --- payload shape ---------------------------------------------------------


def test_video_clips_and_frames_are_collected(fixture):
    payload = collect_payload(VIDEO_ID, fixture)
    assert payload.video["video_id"] == VIDEO_ID
    assert len(payload.clips) == 2
    assert len(payload.frames) == 4
    assert all(f["clip_uid"].startswith(f"{VIDEO_ID}:") for f in payload.frames)


def test_winning_attribute_is_denormalized_onto_the_object(fixture):
    """The indexed property that makes "all white sedans" one seek."""
    payload = collect_payload(VIDEO_ID, fixture)
    obj = payload.objects[0]
    assert obj["color"] == "white"
    assert obj["color_confidence"] == pytest.approx(0.9)
    assert obj["color_source"] == "agreed"
    assert obj["color_uncertain"] is False
    assert obj["label"] == "Vehicle"


def test_full_distribution_is_normalized_into_attribute_edges(fixture):
    """Both candidate colours survive as edges, not just the winner -- this is
    what a serialized JSON property could never answer in Cypher."""
    payload = collect_payload(VIDEO_ID, fixture)
    colors = {e["value"]: e for e in payload.attribute_edges if e["type"] == "color"}
    assert set(colors) == {"white", "silver"}
    assert colors["white"]["winner"] is True
    assert colors["silver"]["winner"] is False
    # Clip A (10 votes, 0.8 white) outweighs clip B (5 votes, 0.4 white).
    assert colors["white"]["share"] > colors["silver"]["share"]
    assert sum(e["share"] for e in colors.values()) == pytest.approx(1.0)


def test_best_shot_only_attribute_still_gets_an_edge(fixture):
    """M8 can answer where clip-level voting never did, so the winner may not
    appear in any clip-level distribution."""
    payload = collect_payload(VIDEO_ID, fixture)
    makes = [e for e in payload.attribute_edges if e["type"] == "make"]
    assert len(makes) == 1
    assert makes[0]["value"] == "toyota"
    assert makes[0]["winner"] is True
    assert makes[0]["source"] == "best_shot"


def test_sightings_link_objects_to_clips_and_frames(fixture):
    payload = collect_payload(VIDEO_ID, fixture)
    assert {s["track_id"] for s in payload.clip_sightings} == {"1", "7"}
    assert len(payload.frame_sightings) == 4  # two frames in each of two clips
    assert all(len(s["bbox"]) == 4 for s in payload.frame_sightings)


def test_events_produce_roles_and_a_shortcut_edge(fixture):
    payload = collect_payload(VIDEO_ID, fixture)
    assert len(payload.events) == 1
    assert payload.events[0]["type"] == "STOP"
    assert [r["role"] for r in payload.event_roles] == ["subject"]
    assert [e["rel"] for e in payload.shortcut_edges] == ["STOPS_AT"]
    assert payload.shortcut_edges[0]["to_kind"] == "frame"


def test_missing_object_index_is_a_clear_error(fixture):
    outputs = fixture.resolve_path(fixture.paths.outputs_dir)
    (outputs / "global_objects_final" / f"{VIDEO_ID}.json").unlink()
    with pytest.raises(KGBuilderError, match="run `link` first"):
        collect_payload(VIDEO_ID, fixture)


def test_collect_payload_is_deterministic(fixture):
    """Same inputs, same graph -- no set/dict iteration order leaking through."""
    first = collect_payload(VIDEO_ID, fixture)
    second = collect_payload(VIDEO_ID, fixture)
    assert first == second


# --- schema file -----------------------------------------------------------


def test_splitter_strips_comments_before_splitting():
    """A semicolon inside a comment must not tear a statement in half --
    splitting first left the rest of the comment as a syntax error."""
    text = "// note; with a semicolon\nMATCH (n) RETURN n;\n// trailing; comment\n"
    assert split_cypher_statements(text) == ["MATCH (n) RETURN n"]


def test_example_queries_file_holds_exactly_ten_statements():
    path = Path(__file__).resolve().parents[1] / "src" / "graph" / "queries.cypher"
    statements = split_cypher_statements(path.read_text())
    assert len(statements) == 10
    assert all(s.upper().startswith("MATCH") for s in statements)


def test_schema_statements_parse_into_individual_ddl():
    statements = _schema_statements()
    assert len(statements) >= 15
    assert all(s.startswith("CREATE") for s in statements)
    assert any("IS UNIQUE" in s for s in statements)
    # The composite index the headline filter query depends on.
    assert any("o.color, o.vehicle_type" in s for s in statements)
    assert not any("//" in s for s in statements)  # comments stripped
