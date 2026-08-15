"""Tests for M9 rule-based event detection: geometric primitives and each
detector in isolation, on hand-built trajectories."""
from __future__ import annotations

import pytest

from src.graph.event_detector import (
    _Line,
    _Polygon,
    _TrajSample,
    _angular_diff,
    _detect_crosses,
    _detect_lane_changes,
    _detect_overtakes,
    _detect_stop_and_park,
    _detect_turns,
    _mean_heading,
    _point_in_polygon,
    _segments_intersect,
)
from src.utils.config import get_settings


def sample(t: float, x: float, y: float, vx: float, vy: float, frame: str | None = None) -> _TrajSample:
    return _TrajSample(
        t=t,
        wallclock=f"2026-08-15T10:00:{t:05.2f}",
        clip_id="clip_000",
        frame_id=frame or f"f{t:.2f}",
        center=(x, y),
        bbox=(x - 25, y - 25, x + 25, y + 25),
        velocity=(vx, vy),
    )


# --- geometric primitives --------------------------------------------------


def test_angular_diff_wraps_correctly():
    assert _angular_diff(170, -170) == pytest.approx(20)
    assert _angular_diff(-170, 170) == pytest.approx(-20)
    assert _angular_diff(0, 90) == pytest.approx(90)
    assert _angular_diff(0, 0) == pytest.approx(0)


def test_mean_heading_uses_circular_mean():
    """Headings near +/-180 must not cancel to a bogus 0."""
    trajectory = [sample(0, 0, 0, -100, 1), sample(1, -100, 0, -100, -1)]
    heading = _mean_heading(trajectory, 0, 1)
    assert abs(heading) > 170  # near 180, not near 0


def test_segments_intersect():
    assert _segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not _segments_intersect((0, 0), (10, 10), (20, 20), (30, 30))


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _point_in_polygon((5, 5), square)
    assert not _point_in_polygon((15, 5), square)


# --- STOP / PARK ------------------------------------------------------------


def test_stop_detected_for_sustained_near_zero_speed():
    settings = get_settings()
    trajectory = [
        sample(0, 0, 0, 100, 0),
        sample(1, 100, 0, 0.1, 0),
        sample(2, 100.1, 0, 0.1, 0),
        sample(3, 100.2, 0, 0.1, 0),
        sample(4, 200, 0, 100, 0),  # resumes moving
    ]
    events = _detect_stop_and_park("obj_1", trajectory, settings)
    assert len(events) == 1
    assert events[0].type == "STOP"
    assert events[0].start_time == trajectory[1].wallclock
    assert events[0].end_time == trajectory[3].wallclock


def test_park_detected_when_stopped_through_end_of_observation():
    settings = get_settings()
    trajectory = [
        sample(0, 0, 0, 100, 0),
        sample(1, 100, 0, 0.0, 0),
        sample(2, 100, 0, 0.0, 0),
        sample(3, 100, 0, 0.0, 0),
    ]
    events = _detect_stop_and_park("obj_1", trajectory, settings)
    assert len(events) == 1
    assert events[0].type == "PARK"


def test_brief_stop_below_minimum_duration_is_not_an_event():
    settings = get_settings()
    trajectory = [sample(0, 0, 0, 100, 0), sample(0.5, 50, 0, 0.0, 0), sample(1.0, 100, 0, 100, 0)]
    assert _detect_stop_and_park("obj_1", trajectory, settings) == []


def test_never_stopped_produces_no_events():
    settings = get_settings()
    trajectory = [sample(t, t * 100, 0, 100, 0) for t in range(5)]
    assert _detect_stop_and_park("obj_1", trajectory, settings) == []


# --- TURN --------------------------------------------------------------


def test_turn_detected_on_sustained_heading_change():
    settings = get_settings()
    # Heading eases from 0 deg (east) through 20 deg (still under the 45 deg
    # gate) to 90 deg (south) over 2s, so the match can only close at j=2 --
    # the scan takes the smallest qualifying window, so a heading change big
    # enough to already pass at j=1 would report 20 deg, not 90.
    trajectory = [
        sample(0.0, 0, 0, 100, 0),
        sample(1.0, 100, 0, 94, 34),
        sample(2.0, 170, 70, 0, 100),
    ]
    events = _detect_turns("obj_1", trajectory, settings)
    assert len(events) == 1
    assert events[0].type == "TURN"
    assert events[0].metadata["turn_direction"] == "right"
    assert float(events[0].metadata["angle_deg"]) == pytest.approx(90, abs=1)


def test_turn_reports_the_earliest_qualifying_window():
    """Determinism: the scan takes the smallest window that clears the gate,
    not the largest heading change available."""
    settings = get_settings()
    trajectory = [
        sample(0.0, 0, 0, 100, 0),
        sample(1.0, 100, 0, 0, 100),  # already a 90 deg change here
        sample(2.0, 100, 100, -100, 0),  # 180 deg change from start
    ]
    events = _detect_turns("obj_1", trajectory, settings)
    assert len(events) == 1
    assert float(events[0].metadata["angle_deg"]) == pytest.approx(90, abs=1)
    assert events[0].end_time == trajectory[1].wallclock


def test_straight_travel_produces_no_turn():
    settings = get_settings()
    trajectory = [sample(t, t * 100, 0, 100, 0) for t in range(5)]
    assert _detect_turns("obj_1", trajectory, settings) == []


def test_turn_direction_sign_for_left_turn():
    settings = get_settings()
    trajectory = [
        sample(0.0, 0, 0, 100, 0),
        sample(1.0, 100, 0, 70, -70),
        sample(2.0, 170, -70, 0, -100),
    ]
    events = _detect_turns("obj_1", trajectory, settings)
    assert len(events) == 1
    assert events[0].metadata["turn_direction"] == "left"


# --- LANE_CHANGE ---------------------------------------------------------


def test_lane_change_detected_for_lateral_shift_without_heading_change():
    settings = get_settings()
    # A real lane change: heading matches at both window endpoints (pointing
    # east), but the object steers briefly off-axis in between, ending up
    # laterally offset. The path (an S-curve) is not just "constant heading
    # times elapsed time" -- if it were, displacement would be exactly along
    # the heading and there could be no lateral component to detect.
    centers = [(0, 0), (50, 10), (100, 30), (150, 50), (200, 60), (250, 60)]
    velocities = [(50, 0), (50, 20), (50, 20), (50, 20), (50, -20), (50, 0)]
    trajectory = [
        sample(t, cx, cy, vx, vy) for t, ((cx, cy), (vx, vy)) in enumerate(zip(centers, velocities))
    ]
    events = _detect_lane_changes("obj_1", trajectory, settings)
    assert len(events) == 1
    assert events[0].type == "LANE_CHANGE"
    assert float(events[0].metadata["lateral_px"]) == pytest.approx(60, abs=1)


def test_turn_and_lane_change_are_mutually_exclusive_on_the_same_window():
    """A large heading change must not also register as a lane change."""
    settings = get_settings()
    trajectory = [
        sample(0.0, 0, 0, 100, 0),
        sample(1.0, 100, 0, 70, 70),
        sample(2.0, 170, 70, 0, 100),
    ]
    assert _detect_turns("obj_1", trajectory, settings) != []
    assert _detect_lane_changes("obj_1", trajectory, settings) == []


def test_small_lateral_drift_below_threshold_is_not_a_lane_change():
    settings = get_settings()
    trajectory = [sample(t, t * 50, min(t * 2, 5), 50, 0) for t in range(6)]
    assert _detect_lane_changes("obj_1", trajectory, settings) == []


# --- CROSSES --------------------------------------------------------------


def test_crosses_line_detected_on_segment_intersection():
    settings = get_settings()
    trajectory = [sample(0, 0, 50, 100, 0), sample(1, 100, 50, 100, 0)]
    line = _Line(id="stop_line", p1=(50, 0), p2=(50, 100))
    events = _detect_crosses("obj_1", trajectory, [line], [], settings)
    assert len(events) == 1
    assert events[0].metadata["region_id"] == "stop_line"


def test_no_crossing_produces_no_event():
    settings = get_settings()
    trajectory = [sample(0, 0, 50, 100, 0), sample(1, 40, 50, 100, 0)]
    line = _Line(id="stop_line", p1=(50, 0), p2=(50, 100))
    assert _detect_crosses("obj_1", trajectory, [line], [], settings) == []


def test_crosses_region_records_enter_and_exit():
    settings = get_settings()
    polygon = _Polygon(id="intersection", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
    trajectory = [
        sample(0, -50, 50, 100, 0),
        sample(1, 50, 50, 100, 0),   # entering
        sample(2, 150, 50, 100, 0),  # exiting
    ]
    events = _detect_crosses("obj_1", trajectory, [], [polygon], settings)
    assert [e.metadata["direction"] for e in events] == ["enter", "exit"]


# --- OVERTAKE --------------------------------------------------------------


def test_overtake_detected_when_a_passes_b():
    settings = get_settings()
    # Both travel east, offset laterally by 20px. A is faster and starts
    # behind B; by t=6 it has pulled ahead and stays ahead.
    traj_a = [sample(t, t * 150, 20, 150, 0) for t in range(7)]  # x: 0 -> 900
    traj_b = [sample(t, t * 100 + 250, 0, 100, 0) for t in range(7)]  # x: 250 -> 850
    events = _detect_overtakes("A", traj_a, "B", traj_b, settings)
    assert len(events) == 1
    assert events[0].type == "OVERTAKE"
    assert events[0].subject_id == "A"
    assert events[0].object_id == "B"


def test_no_overtake_when_no_lateral_displacement():
    """Same line, A simply catches up and passes directly behind/ahead with
    zero lateral offset -- roadmap requires lateral displacement to be present."""
    settings = get_settings()
    traj_a = [sample(t, t * 150, 0, 150, 0) for t in range(6)]
    traj_b = [sample(t, t * 100 + 250, 0, 100, 0) for t in range(6)]
    assert _detect_overtakes("A", traj_a, "B", traj_b, settings) == []


def test_no_overtake_when_directions_differ():
    settings = get_settings()
    traj_a = [sample(t, t * 150, 20, 150, 0) for t in range(6)]  # eastbound
    traj_b = [sample(t, 500 - t * 100, 0, -100, 0) for t in range(6)]  # westbound
    assert _detect_overtakes("A", traj_a, "B", traj_b, settings) == []


def test_no_overtake_when_never_crosses():
    settings = get_settings()
    traj_a = [sample(t, t * 100, 20, 100, 0) for t in range(6)]
    traj_b = [sample(t, t * 100 + 500, 0, 100, 0) for t in range(6)]  # always far ahead
    assert _detect_overtakes("A", traj_a, "B", traj_b, settings) == []
