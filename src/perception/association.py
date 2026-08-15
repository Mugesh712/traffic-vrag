"""M4 — Deterministic, appearance-gated intra-clip association (Contribution #3).

Repairs ByteTrack fragmentation: one physical object that occlusion split into
several track_ids is stitched back into a single track.

SCOPE: merge-only. This repairs *fragmentation* (one object -> many ids). It does
not split an id that drifted between two objects (an ID switch proper), which
would need intra-track appearance-discontinuity detection instead.

METHOD. Every candidate link must clear four HARD, conjunctive gates:

    G1 TEMPORAL     strictly disjoint in time, gap <= max_gap_sec
    G2 CLASS        identical object class
    G3 MOTION       constant-velocity prediction lands near the successor's entry
                    (stationary pairs use IoU instead — a parked car has no
                    velocity to extrapolate, so prediction is meaningless there)
    G4 APPEARANCE   ReID cosine similarity >= tau_app

Because the gates are conjunctive, the surviving set is INDEPENDENT of their
order. The order is fixed for three other reasons: cost (integer compare ->
string compare -> ~10 flops -> 512-d dot product), selectivity (G1 alone cuts
the O(n^2) pair space down to temporally adjacent pairs), and rejection
attribution (the ablation logs each rejection against the FIRST gate to fail,
which is only interpretable against a documented order).

Appearance is deliberately last. It is the least discriminative signal in
traffic surveillance — a fleet of identical white sedans defeats it — and the
most sensitive to blur and lighting. Geometry proposes; appearance only vetoes.

DETERMINISM. Same input always yields the same output: tracks are put in a
canonical order, candidates are enumerated and ranked with total tie-breaking
orders, and nothing consults a hash ordering or an RNG.

Output: data/outputs/tracks_associated/<clip_id>.json (ClipAssociatedTracks)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.manifest import load_clip_frame_index
from src.utils.schemas import (
    ClipAssociatedTracks,
    ClipTracks,
    MergeLogEntry,
    RejectedLinkEntry,
    Track,
)

logger = get_logger(__name__)

# Gate evaluation order. Documented here because rejection attribution — and so
# the paper's ablation table — is only meaningful relative to a fixed order.
GATE_ORDER = ("temporal", "class", "motion", "appearance")


class AssociationError(RuntimeError):
    pass


def _id_sort_key(track_id: str) -> tuple[int, int, str]:
    """Total order over track ids that sorts numeric ids numerically ("2" before
    "10") while still totally ordering non-numeric ones."""
    return (0, int(track_id), "") if track_id.isdigit() else (1, 0, track_id)


@dataclass
class _TrackSummary:
    """Endpoint state of one fragment — all the association gates need."""

    track: Track
    track_id: str
    cls: str
    first_idx: int  # position in the clip's ordered sampled-frame list
    last_idx: int
    first_ts: float
    last_ts: float
    head_bbox: tuple[float, float, float, float]
    tail_bbox: tuple[float, float, float, float]
    head_center: tuple[float, float]
    tail_center: tuple[float, float]
    exit_velocity: tuple[float, float]
    stationary: bool
    embedding: np.ndarray
    n_frames: int

    @property
    def sort_key(self) -> tuple[int, tuple[int, int, str]]:
        return (self.first_idx, _id_sort_key(self.track_id))


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _diagonal(bbox: tuple[float, ...]) -> float:
    w, h = max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])
    return float(np.hypot(w, h))


def _summarize(
    track: Track,
    frame_positions: dict[str, int],
    frame_timestamps: dict[str, float],
    settings: PipelineSettings,
) -> _TrackSummary:
    centers = np.asarray(track.centers, dtype=float)

    # Positional variance decides stationarity. A parked car's centre jitters by
    # a few px of detector noise; a moving one sweeps the frame.
    if len(centers) >= 2:
        variance = float(centers[:, 0].var() + centers[:, 1].var())
    else:
        variance = 0.0
    stationary = variance < settings.association.stationary_variance_threshold

    # Exit velocity is averaged over a short window: a single frame-to-frame
    # difference is dominated by bbox jitter, which would wreck the prediction.
    window = max(1, settings.association.velocity_window)
    tail_velocities = [v for v in track.velocity[1:][-window:]] or [(0.0, 0.0)]
    exit_velocity = (
        float(np.mean([v[0] for v in tail_velocities])),
        float(np.mean([v[1] for v in tail_velocities])),
    )

    embedding = np.asarray(track.embedding, dtype=float)
    norm = float(np.linalg.norm(embedding))
    if norm > 1e-12:
        embedding = embedding / norm

    first_frame, last_frame = track.frames[0], track.frames[-1]
    return _TrackSummary(
        track=track,
        track_id=track.track_id,
        cls=track.cls,
        first_idx=frame_positions[first_frame],
        last_idx=frame_positions[last_frame],
        first_ts=frame_timestamps[first_frame],
        last_ts=frame_timestamps[last_frame],
        head_bbox=tuple(track.bboxes[0]),
        tail_bbox=tuple(track.bboxes[-1]),
        head_center=tuple(track.centers[0]),
        tail_center=tuple(track.centers[-1]),
        exit_velocity=exit_velocity,
        stationary=stationary,
        embedding=embedding,
        n_frames=len(track.frames),
    )


def _evaluate_gates(
    a: _TrackSummary, b: _TrackSummary, settings: PipelineSettings
) -> tuple[str | None, dict[str, float]]:
    """Run the gates in GATE_ORDER. Returns (first failing gate or None, scores).

    Scores for gates after the failing one are not computed — they are not
    needed, and skipping them is the point of the ordering.
    """
    cfg = settings.association
    scores: dict[str, float] = {}

    # G1 TEMPORAL — strict disjointness plus a bounded gap. Cheapest and most
    # selective: it alone reduces the pair space to temporally adjacent pairs.
    gap_sec = b.first_ts - a.last_ts
    scores["temporal_gap_sec"] = gap_sec
    if b.first_idx <= a.last_idx or gap_sec <= 0 or gap_sec > cfg.max_gap_sec:
        return "temporal", scores

    # G2 CLASS — a car does not become a truck.
    scores["class_match"] = 1.0 if a.cls == b.cls else 0.0
    if a.cls != b.cls:
        return "class", scores

    # G3 MOTION — stationary pairs are matched by overlap, moving pairs by
    # constant-velocity extrapolation normalized by object size (so the same
    # tolerance holds for near and far objects).
    if a.stationary and b.stationary:
        iou = _iou(a.tail_bbox, b.head_bbox)
        scores["motion_iou"] = iou
        scores["motion"] = iou
        if iou < cfg.stationary_iou_threshold:
            return "motion", scores
    else:
        dt = gap_sec
        predicted = (
            a.tail_center[0] + a.exit_velocity[0] * dt,
            a.tail_center[1] + a.exit_velocity[1] * dt,
        )
        error_px = float(np.hypot(predicted[0] - b.head_center[0], predicted[1] - b.head_center[1]))
        diag = _diagonal(a.tail_bbox) or 1.0
        error = error_px / diag
        scores["motion_error"] = error
        scores["motion"] = 1.0 / (1.0 + error)
        if error > cfg.motion_tolerance:
            return "motion", scores

    # G4 APPEARANCE — the veto. Most expensive (512-d dot product) and least
    # trustworthy, so it runs last and only confirms what geometry proposed.
    cosine = float(np.dot(a.embedding, b.embedding)) if a.embedding.size and b.embedding.size else 0.0
    scores["appearance"] = cosine
    if cosine < cfg.appearance_similarity_threshold:
        return "appearance", scores

    return None, scores


def _combined_score(scores: dict[str, float], settings: PipelineSettings) -> float:
    """Rank surviving candidates. Ranking only decides which of several admissible
    links wins a contested endpoint; it can never admit a gated-out pair."""
    w = settings.association.score_weights
    temporal_closeness = 1.0 - (scores["temporal_gap_sec"] / settings.association.max_gap_sec)
    return (
        w.appearance * scores.get("appearance", 0.0)
        + w.motion * scores.get("motion", 0.0)
        + w.temporal * temporal_closeness
    )


@dataclass
class _UnionFind:
    parent: dict[str, str] = field(default_factory=dict)

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(b)] = self.find(a)


def _merge_tracks(
    members: list[_TrackSummary], frame_timestamps: dict[str, float], top_k: int
) -> Track:
    """Rebuild one track from its fragments, in temporal order."""
    members = sorted(members, key=lambda m: m.sort_key)

    frames: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []
    centers: list[tuple[float, float]] = []
    for m in members:
        frames.extend(m.track.frames)
        bboxes.extend(tuple(b) for b in m.track.bboxes)
        centers.extend(tuple(c) for c in m.track.centers)

    # Velocity is recomputed across the stitched trajectory rather than
    # concatenated, so the frames spanning a repaired gap get a real value.
    velocity: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(1, len(centers)):
        dt = frame_timestamps[frames[i]] - frame_timestamps[frames[i - 1]]
        if dt <= 0:
            velocity.append((0.0, 0.0))
            continue
        velocity.append(((centers[i][0] - centers[i - 1][0]) / dt, (centers[i][1] - centers[i - 1][1]) / dt))

    dominant_direction_deg = None
    if len(centers) >= 2:
        dx, dy = centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]
        if dx != 0.0 or dy != 0.0:
            dominant_direction_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Frame-count weighted so a 40-frame fragment outweighs a 2-frame one.
    embedding: list[float] = []
    weighted = [(m.embedding, m.n_frames) for m in members if m.embedding.size]
    if weighted:
        stacked = np.sum([e * n for e, n in weighted], axis=0)
        norm = float(np.linalg.norm(stacked))
        if norm > 1e-12:
            embedding = (stacked / norm).tolist()

    # Best shots are re-ranked across the whole merged object, which is the
    # point of storing per-crop scores in M3.
    scored_crops: list[tuple[float, str]] = []
    for m in members:
        scores = m.track.best_shot_scores or [0.0] * len(m.track.best_shot_crops)
        scored_crops.extend(zip(scores, m.track.best_shot_crops))
    scored_crops.sort(key=lambda t: (-t[0], t[1]))
    top_crops = scored_crops[:top_k]

    return Track(
        track_id=members[0].track_id,  # earliest fragment's id survives
        **{"class": members[0].cls},
        frames=frames,
        bboxes=bboxes,
        centers=centers,
        velocity=velocity,
        dominant_direction_deg=dominant_direction_deg,
        embedding=embedding,
        best_shot_crops=[path for _, path in top_crops],
        best_shot_scores=[score for score, _ in top_crops],
    )


def associate_clip(clip_id: str, settings: PipelineSettings | None = None) -> ClipAssociatedTracks:
    settings = settings or get_settings()

    tracks_path = settings.resolve_path(settings.paths.outputs_dir) / "tracks" / f"{clip_id}.json"
    if not tracks_path.exists():
        raise AssociationError(f"No tracks found for {clip_id} at {tracks_path}; run `track` first.")
    clip_tracks = ClipTracks.model_validate_json(tracks_path.read_text())

    _, frame_timestamps, _ = load_clip_frame_index(clip_id, settings)
    frame_positions = {frame_id: i for i, frame_id in enumerate(sorted(frame_timestamps))}

    summaries = [
        _summarize(t, frame_positions, frame_timestamps, settings)
        for t in clip_tracks.tracks
        if t.frames
    ]
    summaries.sort(key=lambda s: s.sort_key)  # canonical order

    logger.info(
        "associate_clip: clip_id=%s n_tracks=%d max_gap_sec=%.1f tau_app=%.2f",
        clip_id,
        len(summaries),
        settings.association.max_gap_sec,
        settings.association.appearance_similarity_threshold,
    )

    # --- Candidate generation + gating -------------------------------------
    candidates: list[tuple[float, _TrackSummary, _TrackSummary, dict[str, float]]] = []
    rejected: list[RejectedLinkEntry] = []

    for i, a in enumerate(summaries):
        for b in summaries[i + 1 :]:
            failed_gate, scores = _evaluate_gates(a, b, settings)
            if failed_gate is None:
                candidates.append((_combined_score(scores, settings), a, b, scores))
            elif failed_gate != "temporal":
                # Temporal failures are the overwhelming majority and carry no
                # information; logging them would swamp the ablation file.
                rejected.append(
                    RejectedLinkEntry(
                        from_track_id=a.track_id,
                        to_track_id=b.track_id,
                        failed_gate=failed_gate,
                        gate_scores=scores,
                    )
                )

    # --- Deterministic best-first matching ---------------------------------
    # Each fragment may take at most one successor and one predecessor, so a
    # repaired object is a temporally ordered chain, never a tree. Ties are
    # broken by canonical order, so the result never depends on enumeration
    # order or floating-point ordering accidents.
    candidates.sort(key=lambda c: (-c[0], c[1].sort_key, c[2].sort_key))

    has_successor: set[str] = set()
    has_predecessor: set[str] = set()
    uf = _UnionFind()
    accepted: list[tuple[_TrackSummary, _TrackSummary, dict[str, float], float]] = []
    suppressed: list[RejectedLinkEntry] = []

    for score, a, b, scores in candidates:
        if a.track_id in has_successor or b.track_id in has_predecessor:
            # Admissible, but a higher-scoring link already claimed an endpoint.
            # Logged separately: this is the matcher's doing, not a gate's.
            suppressed.append(
                RejectedLinkEntry(
                    from_track_id=a.track_id,
                    to_track_id=b.track_id,
                    failed_gate="assignment_conflict",
                    gate_scores={**scores, "combined_score": score},
                )
            )
            continue
        has_successor.add(a.track_id)
        has_predecessor.add(b.track_id)
        uf.union(a.track_id, b.track_id)
        accepted.append((a, b, scores, score))

    # --- Materialize merged tracks -----------------------------------------
    by_id = {s.track_id: s for s in summaries}
    components: dict[str, list[_TrackSummary]] = {}
    for s in summaries:
        components.setdefault(uf.find(s.track_id), []).append(s)

    merged_tracks: list[Track] = []
    surviving_id_of: dict[str, str] = {}
    for root in sorted(components, key=lambda r: by_id[r].sort_key):
        members = components[root]
        merged = _merge_tracks(members, frame_timestamps, settings.track.embedding_top_k)
        merged_tracks.append(merged)
        for m in members:
            surviving_id_of[m.track_id] = merged.track_id

    merge_log = [
        MergeLogEntry(
            merged_track_ids=[a.track_id, b.track_id],
            into_track_id=surviving_id_of[a.track_id],
            gate_scores={**scores, "combined_score": score},
        )
        for a, b, scores, score in accepted
    ]

    associated = ClipAssociatedTracks(
        clip_id=clip_id,
        tracks=merged_tracks,
        merge_log=merge_log,
        rejected_links=rejected,
        suppressed_links=suppressed,
    )

    output_dir = settings.resolve_path(settings.paths.outputs_dir) / "tracks_associated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}.json"
    output_path.write_text(associated.model_dump_json(indent=2, by_alias=True))

    logger.info(
        "associate_clip: clip_id=%s %d tracks -> %d after %d merges "
        "(%d gated-out, %d suppressed) -> %s",
        clip_id,
        len(summaries),
        len(merged_tracks),
        len(merge_log),
        len(rejected),
        len(suppressed),
        output_path,
    )

    return associated
