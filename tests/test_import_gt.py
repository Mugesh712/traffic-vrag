"""Tests for M16's benchmark-annotation importer.

The property that matters most is frame alignment. Benchmarks annotate every
video frame; this pipeline samples a subset. Importing an annotation for a
frame the system never saw turns it into a guaranteed false negative, so MOTA
would report the sampling rate rather than the tracker's quality -- a wrong
number that looks like a real measurement.
"""
from __future__ import annotations

import json

import pytest

from src.eval.ground_truth import load_ground_truth
from src.eval.import_gt import GroundTruthImportError, import_ground_truth
from src.utils.config import get_settings

VIDEO_ID = "video_test_import"


@pytest.fixture
def fixture(tmp_path):
    """A manifest that sampled only every 2nd frame: 0, 2, 4, 6."""
    settings = get_settings()
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    manifest_path = outputs / "ingest" / f"{VIDEO_ID}_manifest.json"
    gt_path = settings.resolve_path("data/ground_truth") / f"{VIDEO_ID}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        {"clip_id": "c0", "frame_id": f"frame_{i:06d}", "frame_path": f"x/{i}.jpg",
         "video_timestamp_sec": i * 0.08, "wallclock_time": "2026-08-16T08:30:00",
         "fps": 12.5, "resolution": [768, 432]}
        for i in (0, 2, 4, 6)
    ]
    manifest_path.write_text(json.dumps({
        "video_id": VIDEO_ID, "source_path": "x.mp4", "fps": 12.5,
        "resolution": [768, 432], "frame_count": 8,
        "start_wallclock": "2026-08-16T08:30:00", "end_wallclock": "2026-08-16T08:30:01",
        "clips": [{"video_id": VIDEO_ID, "clip_id": "c0", "clip_path": "c.mp4",
                   "start_wallclock": "2026-08-16T08:30:00",
                   "end_wallclock": "2026-08-16T08:30:01", "fps": 12.5,
                   "resolution": [768, 432], "frames": frames}],
    }))

    yield settings, tmp_path, gt_path
    manifest_path.unlink(missing_ok=True)
    gt_path.unlink(missing_ok=True)


def write_mot(tmp_path, lines):
    path = tmp_path / "gt.txt"
    path.write_text("\n".join(lines))
    return path


# --- frame alignment --------------------------------------------------------


def test_annotations_on_unsampled_frames_are_dropped(fixture):
    """The core guard. MOT frames 1..7 (1-indexed) map to video indices 0..6,
    but only 0, 2, 4, 6 were sampled -- the odd ones must not become misses."""
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, [
        "1,1,10,10,50,50,1,3,1",   # -> index 0, sampled   KEEP
        "2,1,11,10,50,50,1,3,1",   # -> index 1, NOT sampled
        "3,1,12,10,50,50,1,3,1",   # -> index 2, sampled   KEEP
        "4,1,13,10,50,50,1,3,1",   # -> index 3, NOT sampled
    ])
    report = import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    assert report.n_boxes_kept == 2
    assert report.n_boxes_dropped_unsampled == 2

    gt = load_ground_truth(VIDEO_ID, settings)
    assert set(gt.tracks[0].boxes) == {"frame_000000", "frame_000002"}


def test_mot_frame_numbers_are_rebased_from_one_to_zero(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["1,1,10,10,50,50,1,3,1"])
    import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    gt = load_ground_truth(VIDEO_ID, settings)
    # MOT frame 1 is the video's first frame, which this pipeline calls 000000.
    assert "frame_000000" in gt.tracks[0].boxes


def test_frame_offset_is_configurable_for_zero_indexed_sources(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["0,1,10,10,50,50,1,3,1"])
    import_ground_truth(VIDEO_ID, source, "mot", settings, frame_offset=0, overwrite=True)
    gt = load_ground_truth(VIDEO_ID, settings)
    assert "frame_000000" in gt.tracks[0].boxes


def test_a_track_only_on_unsampled_frames_disappears_and_is_reported(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, [
        "1,1,10,10,50,50,1,3,1",   # index 0, kept
        "2,2,99,99,20,20,1,3,1",   # index 1, never sampled -> track 2 vanishes
    ])
    report = import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    assert report.n_tracks == 1
    assert report.dropped_track_ids == ["gt_2"]


# --- box conversion ---------------------------------------------------------


def test_xywh_is_converted_to_xyxy(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["1,1,10,20,50,60,1,3,1"])
    import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    gt = load_ground_truth(VIDEO_ID, settings)
    assert gt.tracks[0].boxes["frame_000000"] == [10.0, 20.0, 60.0, 80.0]


# --- class handling ---------------------------------------------------------


def test_undetectable_classes_are_dropped(fixture):
    """Importing a class the detector can never emit would guarantee misses."""
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, [
        "1,1,10,10,50,50,1,3,1",    # car -> kept
        "1,2,10,10,50,50,1,11,1",   # class 11 (unmapped) -> dropped
    ])
    report = import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    assert report.n_boxes_kept == 1
    assert report.n_boxes_dropped_class == 1


def test_zero_confidence_rows_are_ignored(fixture):
    """MOT marks ignore-regions/distractors with conf=0."""
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, [
        "1,1,10,10,50,50,1,3,1",
        "1,2,10,10,50,50,0,3,1",   # conf 0 -> not an object
    ])
    report = import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    assert report.n_tracks == 1


# --- UA-DETRAC XML ----------------------------------------------------------


DETRAC_XML = """<sequence name="MVI_TEST">
  <frame num="1">
    <target_list>
      <target id="1">
        <box left="10" top="20" width="50" height="60"/>
        <attribute vehicle_type="car"/>
      </target>
      <target id="2">
        <box left="100" top="100" width="80" height="90"/>
        <attribute vehicle_type="van"/>
      </target>
    </target_list>
  </frame>
  <frame num="2">
    <target_list>
      <target id="3">
        <box left="5" top="5" width="30" height="30"/>
        <attribute vehicle_type="bus"/>
      </target>
    </target_list>
  </frame>
</sequence>
"""


def test_detrac_xml_parses_boxes_and_classes(fixture):
    settings, tmp_path, _ = fixture
    source = tmp_path / "MVI_TEST.xml"
    source.write_text(DETRAC_XML)
    import_ground_truth(VIDEO_ID, source, "detrac", settings, overwrite=True)

    gt = load_ground_truth(VIDEO_ID, settings)
    by_id = {t.gt_id: t for t in gt.tracks}
    assert by_id["gt_1"].cls == "car"
    assert by_id["gt_1"].boxes["frame_000000"] == [10.0, 20.0, 60.0, 80.0]
    # A van is a car to the detector, with "van" as its body type.
    assert by_id["gt_2"].cls == "car"


def test_detrac_seeds_vehicle_type_attributes(fixture):
    """UA-DETRAC labels body type, which partially fills attribute ground
    truth -- so colour is the only thing left needing a human."""
    settings, tmp_path, _ = fixture
    source = tmp_path / "MVI_TEST.xml"
    source.write_text(DETRAC_XML)
    report = import_ground_truth(VIDEO_ID, source, "detrac", settings, overwrite=True)

    gt = load_ground_truth(VIDEO_ID, settings)
    attributes = gt.attributes_by_id()
    assert attributes["gt_2"].vehicle_type == "van"
    assert attributes["gt_2"].color is None  # still needs annotating
    assert report.n_attributes >= 1


# --- imported ground truth is immediately usable ----------------------------


def test_imported_ground_truth_has_no_unreviewed_marker(fixture):
    """Unlike the hand-annotation template, this is real human-made truth from
    a published benchmark, so the evaluator must accept it without edits."""
    settings, tmp_path, gt_path = fixture
    source = write_mot(tmp_path, ["1,1,10,10,50,50,1,3,1"])
    import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    assert load_ground_truth(VIDEO_ID, settings) is not None


# --- errors -----------------------------------------------------------------


def test_missing_source_is_a_clear_error(fixture):
    settings, tmp_path, _ = fixture
    with pytest.raises(GroundTruthImportError, match="No annotation file"):
        import_ground_truth(VIDEO_ID, tmp_path / "nope.txt", "mot", settings, overwrite=True)


def test_unknown_format_is_rejected(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["1,1,10,10,50,50,1,3,1"])
    with pytest.raises(GroundTruthImportError, match="Unknown format"):
        import_ground_truth(VIDEO_ID, source, "kitti", settings, overwrite=True)


def test_refuses_to_overwrite_without_permission(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["1,1,10,10,50,50,1,3,1"])
    import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
    with pytest.raises(GroundTruthImportError, match="already exists"):
        import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=False)


def test_malformed_mot_line_names_the_line(fixture):
    settings, tmp_path, _ = fixture
    source = write_mot(tmp_path, ["1,1,10"])
    with pytest.raises(GroundTruthImportError, match="gt.txt:1"):
        import_ground_truth(VIDEO_ID, source, "mot", settings, overwrite=True)
