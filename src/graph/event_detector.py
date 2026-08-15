"""M9 — Rule-based event detection from global object trajectories.

Rules first, LLM later (per the roadmap): every event here is a deterministic
geometric test over a global object's stitched trajectory, explainable and
fast. An optional LLM event-describer is future work layered on top of this,
not a replacement for it.

TRAJECTORY RECONSTRUCTION. M7's master_object_index gives each global object
as a list of (clip_id, track_id) sightings. Each sighting's positions and
velocities live in that clip's tracks_associated file, on that clip's local
timestamp axis -- except M1 computes video_timestamp_sec from the GLOBAL frame
index of the whole video, not a per-clip counter, so timestamps across clips
are already on one shared axis. Stitching a trajectory is therefore just
concatenating each sighting's samples and sorting by time; no offset
arithmetic is needed, and this is verified by a test.

FIVE SINGLE-OBJECT RULES (STOP, PARK, TURN, LANE_CHANGE, CROSSES) and one
PAIRWISE rule (OVERTAKE). See each detector's docstring for its specific test.

REGIONS/LINES are loaded per video_id from configs/regions/<video_id>.yaml so
the same code works across camera angles. A video without a regions file
simply produces no CROSSES events for that video.

Output: data/outputs/events/<video_id>.json (EventLog)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.manifest import load_clip_frame_index
from src.utils.schemas import ClipAssociatedTracks, Event, EventLog, MasterObjectIndex

logger = get_logger(__name__)


class EventDetectorError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------


@dataclass
class _TrajSample:
    t: float  # video-global timestamp, seconds
    wallclock: str
    clip_id: str
    frame_id: str
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]  # px/sec

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def heading_deg(self) -> float | None:
        if self.speed < 1e-9:
            return None
        return float(np.degrees(np.arctan2(self.velocity[1], self.velocity[0])))


def _build_trajectory(
    object_sightings: list[tuple[str, str]],  # (clip_id, track_id)
    settings: PipelineSettings,
    clip_tracks_cache: dict[str, ClipAssociatedTracks],
    frame_index_cache: dict[str, tuple[dict[str, str], dict[str, float], dict[str, str]]],
) -> list[_TrajSample]:
    samples: list[_TrajSample] = []

    for clip_id, track_id in object_sightings:
        if clip_id not in clip_tracks_cache:
            path = settings.resolve_path(settings.paths.outputs_dir) / "tracks_associated" / f"{clip_id}.json"
            if not path.exists():
                logger.warning("event_detector: no associated tracks for %s, skipping sighting", clip_id)
                continue
            clip_tracks_cache[clip_id] = ClipAssociatedTracks.model_validate_json(path.read_text())
        if clip_id not in frame_index_cache:
            frame_index_cache[clip_id] = load_clip_frame_index(clip_id, settings)
        _, frame_ts, frame_wallclock = frame_index_cache[clip_id]

        track = next((t for t in clip_tracks_cache[clip_id].tracks if t.track_id == track_id), None)
        if track is None:
            continue

        for frame_id, bbox, center, velocity in zip(
            track.frames, track.bboxes, track.centers, track.velocity
        ):
            samples.append(
                _TrajSample(
                    t=frame_ts[frame_id],
                    wallclock=frame_wallclock[frame_id],
                    clip_id=clip_id,
                    frame_id=frame_id,
                    center=tuple(center),
                    bbox=tuple(bbox),
                    velocity=tuple(velocity),
                )
            )

    samples.sort(key=lambda s: s.t)
    return samples


def _evidence_frames(trajectory: list[_TrajSample], i: int, j: int, max_frames: int) -> list[str]:
    """Up to max_frames frame ids, evenly spread across [i, j]."""
    if j <= i:
        return [trajectory[i].frame_id]
    n = min(max_frames, j - i + 1)
    positions = np.linspace(i, j, n)
    indices = sorted(set(int(round(p)) for p in positions))
    return [trajectory[idx].frame_id for idx in indices]


def _angular_diff(a: float, b: float) -> float:
    """Signed difference b - a, wrapped to (-180, 180]."""
    return ((b - a + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# STOP / PARK
# ---------------------------------------------------------------------------


def _detect_stop_and_park(
    global_id: str, trajectory: list[_TrajSample], settings: PipelineSettings
) -> list[Event]:
    """A maximal run of near-zero speed, sustained for stop_min_duration_sec.

    PARK is not a separate test: it is a STOP run whose end is the object's
    last observed sample, i.e. it never resumed moving before track was lost.
    """
    cfg = settings.events
    events: list[Event] = []
    n = len(trajectory)
    i = 0

    while i < n:
        if trajectory[i].speed >= cfg.stop_speed_threshold_px_s:
            i += 1
            continue
        j = i
        while j + 1 < n and trajectory[j + 1].speed < cfg.stop_speed_threshold_px_s:
            j += 1

        duration = trajectory[j].t - trajectory[i].t
        if duration >= cfg.stop_min_duration_sec:
            avg_speed = float(np.mean([s.speed for s in trajectory[i : j + 1]]))
            speed_confidence = np.clip(1.0 - avg_speed / cfg.stop_speed_threshold_px_s, 0.0, 1.0)
            duration_confidence = np.clip(duration / (2 * cfg.stop_min_duration_sec), 0.0, 1.0)
            confidence = float((speed_confidence + duration_confidence) / 2.0)

            event_type = "PARK" if j == n - 1 else "STOP"
            events.append(
                Event(
                    event_id="",  # assigned once, after all events are collected
                    type=event_type,
                    subject_id=global_id,
                    start_time=trajectory[i].wallclock,
                    end_time=trajectory[j].wallclock,
                    confidence=confidence,
                    evidence_frames=_evidence_frames(trajectory, i, j, cfg.max_evidence_frames),
                )
            )
        i = j + 1

    return events


# ---------------------------------------------------------------------------
# TURN / LANE_CHANGE — shared sliding-window scan
# ---------------------------------------------------------------------------


def _scan_windows(
    trajectory: list[_TrajSample],
    moving_threshold: float,
    min_window_sec: float,
    max_window_sec: float,
    test_fn,
) -> list[tuple[int, int]]:
    """For each start index with a defined heading, find the smallest end index
    within [min_window_sec, max_window_sec] for which test_fn(i, j) holds.
    On a match, resume scanning after j so matches never overlap; on no match,
    advance by one. Deterministic: earliest-start, smallest-window wins.
    """
    n = len(trajectory)
    matches: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if trajectory[i].speed < moving_threshold:
            i += 1
            continue
        found = None
        j = i + 1
        while j < n:
            dt = trajectory[j].t - trajectory[i].t
            if dt > max_window_sec:
                break
            if dt >= min_window_sec and trajectory[j].speed >= moving_threshold and test_fn(i, j):
                found = j
                break
            j += 1
        if found is not None:
            matches.append((i, found))
            i = found + 1
        else:
            i += 1
    return matches


def _detect_turns(
    global_id: str, trajectory: list[_TrajSample], settings: PipelineSettings
) -> list[Event]:
    """Heading change exceeding turn_min_degrees, sustained across a bounded window."""
    cfg = settings.events

    def is_turn(i: int, j: int) -> bool:
        return abs(_angular_diff(trajectory[i].heading_deg, trajectory[j].heading_deg)) >= cfg.turn_min_degrees

    matches = _scan_windows(
        trajectory, cfg.moving_speed_threshold_px_s, cfg.turn_min_window_sec, cfg.turn_max_window_sec, is_turn
    )

    events = []
    for i, j in matches:
        diff = _angular_diff(trajectory[i].heading_deg, trajectory[j].heading_deg)
        events.append(
            Event(
                event_id="",
                type="TURN",
                subject_id=global_id,
                start_time=trajectory[i].wallclock,
                end_time=trajectory[j].wallclock,
                confidence=float(np.clip(abs(diff) / 180.0, 0.0, 1.0)),
                evidence_frames=_evidence_frames(trajectory, i, j, cfg.max_evidence_frames),
                metadata={"turn_direction": "right" if diff > 0 else "left", "angle_deg": f"{diff:.1f}"},
            )
        )
    return events


def _detect_lane_changes(
    global_id: str, trajectory: list[_TrajSample], settings: PipelineSettings
) -> list[Event]:
    """Lateral displacement exceeding threshold WITHOUT a heading change --
    the complement of TURN. Longitudinal progress must be forward, so pure
    lateral jitter with no real travel doesn't qualify."""
    cfg = settings.events

    def is_lane_change(i: int, j: int) -> bool:
        heading_i = trajectory[i].heading_deg
        if abs(_angular_diff(heading_i, trajectory[j].heading_deg)) >= cfg.lane_change_max_heading_deg:
            return False
        h = np.radians(heading_i)
        h_hat = np.array([np.cos(h), np.sin(h)])
        d = np.array(trajectory[j].center) - np.array(trajectory[i].center)
        longitudinal = float(np.dot(d, h_hat))
        lateral = float(np.linalg.norm(d - longitudinal * h_hat))
        return longitudinal > 0 and lateral >= cfg.lane_change_min_lateral_px

    matches = _scan_windows(
        trajectory,
        cfg.moving_speed_threshold_px_s,
        cfg.lane_change_min_window_sec,
        cfg.lane_change_max_window_sec,
        is_lane_change,
    )

    events = []
    for i, j in matches:
        h = np.radians(trajectory[i].heading_deg)
        h_hat = np.array([np.cos(h), np.sin(h)])
        d = np.array(trajectory[j].center) - np.array(trajectory[i].center)
        longitudinal = float(np.dot(d, h_hat))
        lateral = float(np.linalg.norm(d - longitudinal * h_hat))
        confidence = float(np.clip(lateral / (2 * cfg.lane_change_min_lateral_px), 0.0, 1.0))
        events.append(
            Event(
                event_id="",
                type="LANE_CHANGE",
                subject_id=global_id,
                start_time=trajectory[i].wallclock,
                end_time=trajectory[j].wallclock,
                confidence=confidence,
                evidence_frames=_evidence_frames(trajectory, i, j, cfg.max_evidence_frames),
                metadata={"lateral_px": f"{lateral:.1f}"},
            )
        )
    return events


# ---------------------------------------------------------------------------
# CROSSES — per-video regions/lines
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    id: str
    p1: tuple[float, float]
    p2: tuple[float, float]


@dataclass
class _Polygon:
    id: str
    points: list[tuple[float, float]]


def load_regions(video_id: str, settings: PipelineSettings) -> tuple[list[_Line], list[_Polygon]]:
    """Load configs/regions/<video_id>.yaml. Missing file -> no regions, no
    CROSSES events for this video; that is expected, not an error, since
    regions are optional per-video configuration."""
    path = settings.resolve_path(settings.events.regions_dir) / f"{video_id}.yaml"
    if not path.exists():
        return [], []

    raw = yaml.safe_load(path.read_text()) or {}
    lines = [
        _Line(id=item["id"], p1=tuple(item["points"][0]), p2=tuple(item["points"][1]))
        for item in raw.get("lines", [])
    ]
    polygons = [
        _Polygon(id=item["id"], points=[tuple(p) for p in item["points"]])
        for item in raw.get("regions", [])
    ]
    return lines, polygons


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], p4: tuple[float, float]
) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for k in range(n):
        x1, y1 = polygon[k]
        x2, y2 = polygon[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def _detect_crosses(
    global_id: str,
    trajectory: list[_TrajSample],
    lines: list[_Line],
    polygons: list[_Polygon],
    settings: PipelineSettings,
) -> list[Event]:
    events = []
    for i in range(len(trajectory) - 1):
        a, b = trajectory[i].center, trajectory[i + 1].center

        for line in lines:
            if _segments_intersect(a, b, line.p1, line.p2):
                events.append(
                    Event(
                        event_id="",
                        type="CROSSES",
                        subject_id=global_id,
                        start_time=trajectory[i].wallclock,
                        end_time=trajectory[i + 1].wallclock,
                        confidence=1.0,  # deterministic geometric test
                        evidence_frames=[trajectory[i].frame_id, trajectory[i + 1].frame_id],
                        metadata={"region_id": line.id, "region_type": "line"},
                    )
                )

        for polygon in polygons:
            inside_a = _point_in_polygon(a, polygon.points)
            inside_b = _point_in_polygon(b, polygon.points)
            if inside_a != inside_b:
                events.append(
                    Event(
                        event_id="",
                        type="CROSSES",
                        subject_id=global_id,
                        start_time=trajectory[i].wallclock,
                        end_time=trajectory[i + 1].wallclock,
                        confidence=1.0,
                        evidence_frames=[trajectory[i].frame_id, trajectory[i + 1].frame_id],
                        metadata={
                            "region_id": polygon.id,
                            "region_type": "polygon",
                            "direction": "enter" if inside_b else "exit",
                        },
                    )
                )
    return events


# ---------------------------------------------------------------------------
# OVERTAKE — pairwise
# ---------------------------------------------------------------------------


def _interpolate(trajectory: list[_TrajSample], t: float) -> tuple[float, float] | None:
    """Linear interpolation of center at time t; None if t is outside range."""
    if t < trajectory[0].t or t > trajectory[-1].t:
        return None
    for k in range(len(trajectory) - 1):
        if trajectory[k].t <= t <= trajectory[k + 1].t:
            t0, t1 = trajectory[k].t, trajectory[k + 1].t
            if t1 == t0:
                return trajectory[k].center
            frac = (t - t0) / (t1 - t0)
            c0, c1 = np.array(trajectory[k].center), np.array(trajectory[k + 1].center)
            return tuple(c0 + frac * (c1 - c0))
    return trajectory[-1].center


def _mean_heading(trajectory: list[_TrajSample], t_start: float, t_end: float) -> float | None:
    headings = [s.heading_deg for s in trajectory if t_start <= s.t <= t_end and s.heading_deg is not None]
    if not headings:
        return None
    # Circular mean, so headings near +/-180 don't cancel to a bogus 0.
    radians = np.radians(headings)
    return float(np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def _detect_overtakes(
    id_a: str,
    traj_a: list[_TrajSample],
    id_b: str,
    traj_b: list[_TrajSample],
    settings: PipelineSettings,
) -> list[Event]:
    """A overtakes B: A is behind B, then ahead of B, along their shared
    direction of travel, with the lead sustained and a lateral offset present
    at some point (otherwise this is one object following in another's exact
    path, not a pass). A single longitudinal-difference series captures both
    directions: a negative-to-positive crossing is "A overtakes B"; a
    positive-to-negative crossing at a different time is "B overtakes A".
    """
    cfg = settings.events

    t_start = max(traj_a[0].t, traj_b[0].t)
    t_end = min(traj_a[-1].t, traj_b[-1].t)
    if t_end <= t_start:
        return []

    heading_a = _mean_heading(traj_a, t_start, t_end)
    heading_b = _mean_heading(traj_b, t_start, t_end)
    if heading_a is None or heading_b is None:
        return []
    if abs(_angular_diff(heading_a, heading_b)) > cfg.overtake_max_heading_diff_deg:
        return []

    mean_heading = _mean_heading(traj_a + traj_b, t_start, t_end)
    h = np.radians(mean_heading)
    h_hat = np.array([np.cos(h), np.sin(h)])

    # Shared timeline: every sample time from either trajectory within the
    # overlap, so no real crossing is missed between two irregular sample grids.
    timeline = sorted({s.t for s in traj_a if t_start <= s.t <= t_end} | {s.t for s in traj_b if t_start <= s.t <= t_end})
    if len(timeline) < 2:
        return []

    diffs = []
    laterals = []
    for t in timeline:
        pa, pb = _interpolate(traj_a, t), _interpolate(traj_b, t)
        if pa is None or pb is None:
            diffs.append(None)
            laterals.append(None)
            continue
        d = np.array(pa) - np.array(pb)
        diffs.append(float(np.dot(d, h_hat)))
        laterals.append(float(np.linalg.norm(d - np.dot(d, h_hat) * h_hat)))

    def sample_at(idx: int) -> _TrajSample:
        """Nearest real sample to timeline[idx], for wallclock/frame evidence."""
        t = timeline[idx]
        return min(traj_a + traj_b, key=lambda s: abs(s.t - t))

    # Side convention: >0 is "ahead", <=0 is "behind or tied". Zero belongs to
    # "behind" so an exact tie at one sample and a genuine lead at the next
    # (0 -> positive) still registers as a crossing, instead of silently
    # vanishing between two special-cased comparisons.
    def side(d: float | None) -> int | None:
        return None if d is None else (1 if d > 0 else -1)

    events = []
    k = 1
    n = len(timeline)
    while k < n:
        prev_side, curr_side = side(diffs[k - 1]), side(diffs[k])
        if prev_side is None or curr_side is None or prev_side == curr_side:
            k += 1
            continue

        crossing_idx = k
        sign = curr_side

        # Sustain: the side must hold for overtake_min_sustain_sec, or until
        # the overlap simply runs out of data.
        end_idx = crossing_idx
        while end_idx + 1 < n and side(diffs[end_idx + 1]) == sign:
            end_idx += 1
            if timeline[end_idx] - timeline[crossing_idx] >= cfg.overtake_min_sustain_sec:
                break
        sustained_for = timeline[end_idx] - timeline[crossing_idx]
        reached_data_end = end_idx == n - 1
        if sustained_for < cfg.overtake_min_sustain_sec and not reached_data_end:
            k = end_idx + 1
            continue

        # Lateral displacement present at some point up to the crossing.
        pre_crossing_laterals = [lat for lat in laterals[: crossing_idx + 1] if lat is not None]
        if not pre_crossing_laterals or max(pre_crossing_laterals) < cfg.overtake_min_lateral_px:
            k = end_idx + 1
            continue

        subject, obj = (id_a, id_b) if sign > 0 else (id_b, id_a)
        start_sample = sample_at(0 if crossing_idx == 0 else crossing_idx - 1)
        end_sample = sample_at(end_idx)
        confidence = float(np.clip(abs(diffs[end_idx]) / (2 * max(pre_crossing_laterals)), 0.0, 1.0))

        events.append(
            Event(
                event_id="",
                type="OVERTAKE",
                subject_id=subject,
                object_id=obj,
                start_time=start_sample.wallclock,
                end_time=end_sample.wallclock,
                confidence=confidence,
                evidence_frames=list(dict.fromkeys([start_sample.frame_id, end_sample.frame_id])),
            )
        )
        k = end_idx + 1

    return events


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def detect_events(video_id: str, settings: PipelineSettings | None = None) -> EventLog:
    settings = settings or get_settings()
    outputs_dir = settings.resolve_path(settings.paths.outputs_dir)

    index_path = outputs_dir / "master_object_index" / f"{video_id}.json"
    if not index_path.exists():
        raise EventDetectorError(f"No master object index at {index_path}; run `link` first.")
    master = MasterObjectIndex.model_validate_json(index_path.read_text())

    clip_tracks_cache: dict[str, ClipAssociatedTracks] = {}
    frame_index_cache: dict[str, tuple] = {}

    trajectories: dict[str, list[_TrajSample]] = {}
    for obj in master.objects:
        sightings = [(s.clip_id, s.track_id) for s in obj.sightings]
        trajectories[obj.global_id] = _build_trajectory(
            sightings, settings, clip_tracks_cache, frame_index_cache
        )

    lines, polygons = load_regions(video_id, settings)
    if not lines and not polygons:
        logger.info("detect_events: no regions file for %s; CROSSES will be empty", video_id)

    raw_events: list[Event] = []
    for obj in master.objects:
        trajectory = trajectories[obj.global_id]
        if len(trajectory) < 2:
            continue
        raw_events.extend(_detect_stop_and_park(obj.global_id, trajectory, settings))
        raw_events.extend(_detect_turns(obj.global_id, trajectory, settings))
        raw_events.extend(_detect_lane_changes(obj.global_id, trajectory, settings))
        raw_events.extend(_detect_crosses(obj.global_id, trajectory, lines, polygons, settings))

    object_ids = sorted(trajectories, key=lambda oid: (trajectories[oid][0].t if trajectories[oid] else 0.0, oid))
    for a_idx in range(len(object_ids)):
        for b_idx in range(a_idx + 1, len(object_ids)):
            id_a, id_b = object_ids[a_idx], object_ids[b_idx]
            traj_a, traj_b = trajectories[id_a], trajectories[id_b]
            if len(traj_a) < 2 or len(traj_b) < 2:
                continue
            raw_events.extend(_detect_overtakes(id_a, traj_a, id_b, traj_b, settings))

    # Deterministic id assignment: sort by content, not by detection order,
    # so the same input always yields the same event_ids.
    raw_events.sort(key=lambda e: (e.start_time, e.type, e.subject_id, e.object_id or ""))
    events = [
        e.model_copy(update={"event_id": f"evt_{n:05d}"}) for n, e in enumerate(raw_events, start=1)
    ]

    log = EventLog(events=events)

    output_dir = outputs_dir / "events"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(log.model_dump_json(indent=2))

    counts: dict[str, int] = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    logger.info(
        "detect_events: video_id=%s %d objects -> %d events (%s) -> %s",
        video_id,
        len(master.objects),
        len(events),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
        output_path,
    )

    return log
