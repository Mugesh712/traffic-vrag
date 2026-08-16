"""End-to-end test for M16's runner: with and without ground truth.

Proves the harness actually computes tracking numbers when annotation exists
(the path that cannot be exercised on the real video, which has none), and
that it degrades to explicit "unavailable" rather than zeros when it does not.
"""
from __future__ import annotations

import json

import pytest

from src.eval.report import Table, render_document
from src.eval.runner import evaluate_video, load_outputs
from src.utils.config import get_settings

VIDEO_ID = "video_test_m16"
CLIP_ID = "clip_test_m16"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def fixture():
    settings = get_settings()
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    gt_dir = settings.resolve_path("data/ground_truth")
    written = []

    def track(track_id, frames, boxes):
        return {
            "track_id": track_id, "class": "car", "frames": frames, "bboxes": boxes,
            "centers": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes],
            "velocity": [[0.0, 0.0]] * len(frames), "dominant_direction_deg": None,
            "embedding": [], "best_shot_crops": [], "best_shot_scores": [],
        }

    frames = ["f0", "f1", "f2"]
    box = [0.0, 0.0, 10.0, 10.0]

    manifest_path = outputs / "ingest" / f"{VIDEO_ID}_manifest.json"
    write(manifest_path, {
        "video_id": VIDEO_ID, "source_path": "x.mp4", "fps": 25.0, "resolution": [640, 480],
        "frame_count": 3, "start_wallclock": "2026-08-16T10:00:00", "end_wallclock": "2026-08-16T10:00:03",
        "clips": [{
            "video_id": VIDEO_ID, "clip_id": CLIP_ID, "clip_path": "c.mp4",
            "start_wallclock": "2026-08-16T10:00:00", "end_wallclock": "2026-08-16T10:00:03",
            "fps": 25.0, "resolution": [640, 480],
            "frames": [{"clip_id": CLIP_ID, "frame_id": f, "frame_path": f"x/{f}.jpg",
                        "video_timestamp_sec": i, "wallclock_time": f"2026-08-16T10:00:0{i}",
                        "fps": 25.0, "resolution": [640, 480]} for i, f in enumerate(frames)],
        }],
    })
    written.append(manifest_path)

    # M3: the identity fragments into two ids at f2.
    tracks_path = outputs / "tracks" / f"{CLIP_ID}.json"
    write(tracks_path, {"clip_id": CLIP_ID, "tracks": [
        track("1", ["f0", "f1"], [box, box]), track("2", ["f2"], [box]),
    ]})
    written.append(tracks_path)

    # M4: repaired into a single track covering all three frames.
    assoc_path = outputs / "tracks_associated" / f"{CLIP_ID}.json"
    write(assoc_path, {"clip_id": CLIP_ID, "tracks": [track("1", frames, [box, box, box])],
                       "merge_log": [{"merged_track_ids": ["1", "2"], "into_track_id": "1",
                                      "gate_scores": {}}],
                       "rejected_links": [], "suppressed_links": []})
    written.append(assoc_path)

    canonical_path = outputs / "attributes_canonical" / f"{CLIP_ID}.json"
    write(canonical_path, {"clip_id": CLIP_ID, "tracks": [{"track_id": "1", "votes": [
        {"attribute": "color", "winner": "white", "distribution": {"white": 1.0},
         "uncertain": False, "n_votes": 3, "margin": 1.0, "uncertain_reason": None},
        {"attribute": "make", "winner": None, "distribution": {},
         "uncertain": True, "n_votes": 0, "margin": 0.0, "uncertain_reason": "no_evidence"},
    ]}]})
    written.append(canonical_path)

    master_path = outputs / "master_object_index" / f"{VIDEO_ID}.json"
    write(master_path, {"video_id": VIDEO_ID, "objects": [{
        "global_id": "obj_0001", "class": "car",
        "sightings": [{"clip_id": CLIP_ID, "track_id": "1", "first_seen": "2026-08-16T10:00:00",
                       "last_seen": "2026-08-16T10:00:02", "gate_scores": {}}],
    }], "rejected_links": [], "suppressed_links": []})
    written.append(master_path)

    final_path = outputs / "global_objects_final" / f"{VIDEO_ID}.json"
    write(final_path, {"video_id": VIDEO_ID, "objects": [{
        "global_id": "obj_0001", "class": "car",
        "sightings": [{"clip_id": CLIP_ID, "track_id": "1", "first_seen": "2026-08-16T10:00:00",
                       "last_seen": "2026-08-16T10:00:02", "gate_scores": {}}],
        "attributes": [{"attribute": "color", "value": "white", "confidence": 1.0,
                        "source": "agreed", "uncertain": False}],
        "best_shot_crops": [],
    }], "correction_log": [{"global_id": "obj_0001", "attribute": "color",
                            "clip_level_value": "white", "best_shot_value": "white",
                            "final_value": "white", "outcome": "confirmed"}]})
    written.append(final_path)

    gt_path = gt_dir / f"{VIDEO_ID}.json"
    yield settings, gt_path, written

    for p in written:
        p.unlink(missing_ok=True)
    gt_path.unlink(missing_ok=True)
    for name in (f"{VIDEO_ID}_tables.tex", f"{VIDEO_ID}_metrics.json"):
        (outputs / "eval" / name).unlink(missing_ok=True)


def table_by_label(tables, label):
    return next(t for t in tables if t.label == label)


def test_without_ground_truth_accuracy_tables_are_unavailable_not_zero(fixture):
    settings, gt_path, _ = fixture
    gt_path.unlink(missing_ok=True)

    tables, _ = evaluate_video(VIDEO_ID, settings=settings)
    tracking = table_by_label(tables, "tab:tracking")
    assert tracking.unavailable_reason is not None
    assert tracking.rows == []
    assert "ground-truth" in tracking.unavailable_reason


def test_gt_free_metrics_are_still_reported_without_annotation(fixture):
    settings, gt_path, _ = fixture
    gt_path.unlink(missing_ok=True)

    tables, _ = evaluate_video(VIDEO_ID, settings=settings)
    stages = table_by_label(tables, "tab:stages")
    assert stages.unavailable_reason is None
    assert stages.rows[0]["Identities"] == 2  # M3 fragmented into two
    assert stages.rows[1]["Identities"] == 1  # M4 repaired to one

    corrections = table_by_label(tables, "tab:m8-corrections")
    assert next(r for r in corrections.rows if r["Outcome"] == "Confirmed")["Count"] == 1

    uncertainty = table_by_label(tables, "tab:uncertainty")
    assert {r["Attribute"]: r["Uncertain rate"] for r in uncertainty.rows}["make"] == 1.0


def test_with_ground_truth_tracking_metrics_show_m4_repairing_the_fragment(fixture):
    """The headline comparison: one true identity across three frames, which
    M3 split into two ids and M4 rejoined. Raw ByteTrack should score a lower
    IDF1 than the associated variant, computed from real annotation."""
    settings, gt_path, _ = fixture
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    box = [0.0, 0.0, 10.0, 10.0]
    gt_path.write_text(json.dumps({
        "video_id": VIDEO_ID,
        "tracks": [{"gt_id": "gt_1", "class": "car",
                    "boxes": {"f0": box, "f1": box, "f2": box}}],
        "attributes": [{"gt_id": "gt_1", "color": "white"}],
    }))

    tables, _ = evaluate_video(VIDEO_ID, settings=settings)
    tracking = table_by_label(tables, "tab:tracking")
    assert tracking.unavailable_reason is None

    by_variant = {r["Variant"]: r for r in tracking.rows}
    raw = by_variant["Raw ByteTrack (M3)"]
    associated = by_variant["+ M4 association"]

    # M4 recovers the identity, so IDF1 rises and the switch disappears.
    assert associated["IDF1"] > raw["IDF1"]
    assert associated["IDF1"] == pytest.approx(1.0)
    assert associated["IDSW"] == 0
    assert raw["IDSW"] == 1


def test_with_ground_truth_attribute_accuracy_is_scored(fixture):
    settings, gt_path, _ = fixture
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    box = [0.0, 0.0, 10.0, 10.0]
    gt_path.write_text(json.dumps({
        "video_id": VIDEO_ID,
        "tracks": [{"gt_id": "gt_1", "class": "car", "boxes": {"f0": box, "f1": box, "f2": box}}],
        "attributes": [{"gt_id": "gt_1", "color": "white"}],
    }))

    tables, _ = evaluate_video(VIDEO_ID, settings=settings)
    accuracy = table_by_label(tables, "tab:attr-accuracy")
    assert accuracy.unavailable_reason is None
    colour = next(r for r in accuracy.rows if r["Attribute"] == "color")
    assert colour["Correct"] == 1
    assert colour["Wrong"] == 0


def test_latex_output_is_written_and_parses_as_a_table(fixture):
    settings, gt_path, _ = fixture
    gt_path.unlink(missing_ok=True)
    tables, latex_path = evaluate_video(VIDEO_ID, settings=settings)
    content = latex_path.read_text()
    assert "\\begin{table}" in content
    assert "\\toprule" in content
    assert "booktabs" in content


# --- LaTeX escaping ---------------------------------------------------------


def test_latex_escapes_special_characters():
    table = Table(caption="Rate 50% of items", label="tab:x",
                  columns=["Name"], rows=[{"Name": "a_b & c"}])
    latex = table.to_latex()
    assert r"50\%" in latex
    assert r"a\_b \& c" in latex


def test_unavailable_table_states_the_reason_in_latex():
    table = Table(caption="Tracking", label="tab:t", columns=["A"], rows=[],
                  unavailable_reason="no annotation")
    latex = table.to_latex()
    assert "Not available" in latex
    assert "no annotation" in latex
    assert "\\toprule" not in latex  # no empty table body pretending to be data
