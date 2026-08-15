"""M7 — Semantically-gated cross-clip global linking (Contribution #4).

Assigns a global object id to tracks that are the same physical vehicle seen
across clip boundaries, so Vehicle_12 in clip 3 and Vehicle_47 in clip 4
become one object with one timeline.

FOUR HARD GATES, all must hold:

    G1 CLASS      identical detector class
    G2 SEMANTIC   canonical attributes compatible -- a white sedan cannot
                  become a blue truck
    G3 MOTION     exit state at the end of clip A must plausibly reach the
                  entry position at the start of clip B, given the time gap
    G4 APPEARANCE ReID cosine similarity >= tau_app

IDENTITY vs STATE. The semantic gate compares colour, vehicle_type, make and
model. It deliberately ignores `direction`: those four are identity, direction
is state. A car that turns a corner legitimately changes direction, so gating
on it would reject correct links.

UNCERTAINTY IS PERMISSIVE. An attribute M6 marked uncertain matches anything.
This is where Contribution #2 pays for Contribution #4: a hallucinated
"silver" would block a correct link, whereas an honest "uncertain" steps
aside. Only values the system actually stands behind can veto.

ASSIGNMENT, NOT GREEDY MATCHING. Surviving candidates are scored and solved as
a bipartite assignment with the Hungarian algorithm, so a clip boundary is
resolved globally rather than by whichever pair happened to score highest
first. Each track keeps at most one predecessor and one successor across the
whole video, so a global object is a temporally ordered chain.

Output: data/outputs/master_object_index/<video_id>.json (MasterObjectIndex)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    ClipAssociatedTracks,
    ClipCanonicalAttributes,
    GlobalObject,
    MasterObjectIndex,
    RejectedGlobalLinkEntry,
    Sighting,
    VideoManifest,
)

logger = get_logger(__name__)

# Gate order, documented because rejection attribution depends on it. As in
# M4 the gates are conjunctive, so this order changes cost and blame, never
# which links survive.
GATE_ORDER = ("class", "semantic", "motion", "appearance")

# Identity attributes only. `direction` is state and is excluded on purpose.
SEMANTIC_ATTRIBUTES = ("color", "vehicle_type", "make", "model")

# Finite stand-in for "not linkable" in the cost matrix; scipy needs a dense
# matrix, so infeasible pairs are filled with this and discarded after solving.
_INFEASIBLE_COST = 1e6


class GlobalLinkingError(RuntimeError):
    pass


@dataclass
class _ClipTrackSummary:
    """Endpoint state of one clip-local track, as the cross-clip gates need it."""

    clip_id: str
    track_id: str
    cls: str
    first_ts: float
    last_ts: float
    first_wallclock: str
    last_wallclock: str
    entry_center: tuple[float, float]
    exit_center: tuple[float, float]
    exit_bbox: tuple[float, float, float, float]
    exit_velocity: tuple[float, float]
    embedding: np.ndarray
    # attribute -> (canonical value, uncertain). Uncertain entries match anything.
    attributes: dict[str, tuple[str | None, bool]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.clip_id}:{self.track_id}"

    @property
    def sort_key(self) -> tuple[float, str, str]:
        return (self.first_ts, self.clip_id, self.track_id)


def _diagonal(bbox: tuple[float, ...]) -> float:
    return float(np.hypot(max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])))


def _summarize_clip_tracks(
    clip_tracks: ClipAssociatedTracks,
    canonical: ClipCanonicalAttributes | None,
    frame_ts: dict[str, float],
    frame_wallclock: dict[str, str],
    settings: PipelineSettings,
) -> list[_ClipTrackSummary]:
    attributes_by_track: dict[str, dict[str, tuple[str | None, bool]]] = {}
    if canonical is not None:
        for track in canonical.tracks:
            attributes_by_track[track.track_id] = {
                vote.attribute: (vote.winner, vote.uncertain) for vote in track.votes
            }

    summaries: list[_ClipTrackSummary] = []
    for track in clip_tracks.tracks:
        if not track.frames:
            continue

        window = max(1, settings.linking.velocity_window)
        tail_velocities = list(track.velocity[1:][-window:]) or [(0.0, 0.0)]
        exit_velocity = (
            float(np.mean([v[0] for v in tail_velocities])),
            float(np.mean([v[1] for v in tail_velocities])),
        )

        embedding = np.asarray(track.embedding, dtype=float)
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            embedding = embedding / norm

        first_frame, last_frame = track.frames[0], track.frames[-1]
        summaries.append(
            _ClipTrackSummary(
                clip_id=clip_tracks.clip_id,
                track_id=track.track_id,
                cls=track.cls,
                first_ts=frame_ts[first_frame],
                last_ts=frame_ts[last_frame],
                first_wallclock=frame_wallclock[first_frame],
                last_wallclock=frame_wallclock[last_frame],
                entry_center=tuple(track.centers[0]),
                exit_center=tuple(track.centers[-1]),
                exit_bbox=tuple(track.bboxes[-1]),
                exit_velocity=exit_velocity,
                embedding=embedding,
                attributes=attributes_by_track.get(track.track_id, {}),
            )
        )

    summaries.sort(key=lambda s: s.sort_key)  # canonical order
    return summaries


def _evaluate_gates(
    a: _ClipTrackSummary, b: _ClipTrackSummary, settings: PipelineSettings
) -> tuple[str | None, dict[str, float]]:
    """Run the gates in GATE_ORDER. Returns (first failing gate or None, scores)."""
    cfg = settings.linking
    scores: dict[str, float] = {}

    # G1 CLASS
    scores["class_match"] = 1.0 if a.cls == b.cls else 0.0
    if a.cls != b.cls:
        return "class", scores

    # G2 SEMANTIC — compare only attributes both sides are confident about.
    comparable = 0
    agreed = 0
    if cfg.enable_semantic_gate:
        for attribute in SEMANTIC_ATTRIBUTES:
            a_value, a_uncertain = a.attributes.get(attribute, (None, True))
            b_value, b_uncertain = b.attributes.get(attribute, (None, True))
            if a_value is None or b_value is None or a_uncertain or b_uncertain:
                continue  # uncertain matches anything
            comparable += 1
            if a_value == b_value:
                agreed += 1
        scores["semantic_comparable"] = float(comparable)
        scores["semantic_agreed"] = float(agreed)
        # Any confident disagreement is fatal: a white sedan cannot become a
        # blue truck, however well the motion lines up.
        if comparable and agreed < comparable:
            scores["semantic"] = agreed / comparable
            return "semantic", scores
    scores["semantic"] = (agreed / comparable) if comparable else 1.0

    # G3 MOTION
    dt = b.first_ts - a.last_ts
    scores["gap_sec"] = dt
    if dt <= 0 or dt > cfg.max_gap_sec:
        return "motion", scores
    predicted = (
        a.exit_center[0] + a.exit_velocity[0] * dt,
        a.exit_center[1] + a.exit_velocity[1] * dt,
    )
    error_px = float(np.hypot(predicted[0] - b.entry_center[0], predicted[1] - b.entry_center[1]))
    error = error_px / (_diagonal(a.exit_bbox) or 1.0)
    scores["motion_error"] = error
    scores["motion"] = 1.0 / (1.0 + error)
    if error > cfg.motion_tolerance:
        return "motion", scores

    # G4 APPEARANCE
    if cfg.enable_appearance_gate:
        cosine = float(np.dot(a.embedding, b.embedding)) if a.embedding.size and b.embedding.size else 0.0
        scores["appearance"] = cosine
        if cosine < cfg.appearance_similarity_threshold:
            return "appearance", scores
    else:
        scores["appearance"] = (
            float(np.dot(a.embedding, b.embedding)) if a.embedding.size and b.embedding.size else 0.0
        )

    return None, scores


def _combined_score(scores: dict[str, float], settings: PipelineSettings) -> float:
    w = settings.linking.score_weights
    return (
        w.appearance * scores.get("appearance", 0.0)
        + w.motion * scores.get("motion", 0.0)
        + w.semantic * scores.get("semantic", 0.0)
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


def link_video(video_id: str, settings: PipelineSettings | None = None) -> MasterObjectIndex:
    settings = settings or get_settings()
    outputs_dir = settings.resolve_path(settings.paths.outputs_dir)

    manifest_path = outputs_dir / "ingest" / f"{video_id}_manifest.json"
    if not manifest_path.exists():
        raise GlobalLinkingError(f"No ingest manifest at {manifest_path}; run `ingest` first.")
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())

    # Clip-ordered summaries. Clip order comes from the manifest, not from
    # directory listing, so it follows real time.
    clip_summaries: list[list[_ClipTrackSummary]] = []
    for clip in manifest.clips:
        tracks_path = outputs_dir / "tracks_associated" / f"{clip.clip_id}.json"
        if not tracks_path.exists():
            logger.warning("link_video: no associated tracks for %s, skipping clip", clip.clip_id)
            clip_summaries.append([])
            continue
        clip_tracks = ClipAssociatedTracks.model_validate_json(tracks_path.read_text())

        canonical_path = outputs_dir / "attributes_canonical" / f"{clip.clip_id}.json"
        canonical = None
        if canonical_path.exists():
            canonical = ClipCanonicalAttributes.model_validate_json(canonical_path.read_text())
        elif settings.linking.enable_semantic_gate:
            # Without canonical attributes every attribute reads as uncertain,
            # which makes the semantic gate a no-op. Say so rather than
            # silently running a weaker system.
            logger.warning(
                "link_video: no canonical attributes for %s; semantic gate will not "
                "constrain links for this clip (run `vote` first)",
                clip.clip_id,
            )

        frame_ts = {f.frame_id: f.video_timestamp_sec for f in clip.frames}
        frame_wallclock = {f.frame_id: f.wallclock_time for f in clip.frames}
        clip_summaries.append(
            _summarize_clip_tracks(clip_tracks, canonical, frame_ts, frame_wallclock, settings)
        )

    n_tracks = sum(len(s) for s in clip_summaries)
    logger.info(
        "link_video: video_id=%s n_clips=%d n_tracks=%d max_gap_sec=%.1f semantic_gate=%s",
        video_id,
        len(manifest.clips),
        n_tracks,
        settings.linking.max_gap_sec,
        settings.linking.enable_semantic_gate,
    )

    uf = _UnionFind()
    rejected: list[RejectedGlobalLinkEntry] = []
    suppressed: list[RejectedGlobalLinkEntry] = []
    has_successor: set[str] = set()
    has_predecessor: set[str] = set()
    accepted_scores: dict[str, dict[str, float]] = {}  # successor key -> gate scores
    n_links = 0

    def _entry(a, b, failed_gate, scores) -> RejectedGlobalLinkEntry:
        return RejectedGlobalLinkEntry(
            from_clip_id=a.clip_id,
            from_track_id=a.track_id,
            to_clip_id=b.clip_id,
            to_track_id=b.track_id,
            failed_gate=failed_gate,
            gate_scores=scores,
        )

    for i in range(len(clip_summaries)):
        for j in range(i + 1, min(i + 1 + settings.linking.max_clip_distance, len(clip_summaries))):
            # A track keeps at most one predecessor and one successor across
            # the whole video, so already-claimed endpoints drop out.
            rows = [t for t in clip_summaries[i] if t.key not in has_successor]
            cols = [t for t in clip_summaries[j] if t.key not in has_predecessor]
            if not rows or not cols:
                continue

            cost = np.full((len(rows), len(cols)), _INFEASIBLE_COST, dtype=float)
            feasible = np.zeros((len(rows), len(cols)), dtype=bool)
            pair_scores: dict[tuple[int, int], dict[str, float]] = {}

            for r, a in enumerate(rows):
                for c, b in enumerate(cols):
                    failed_gate, scores = _evaluate_gates(a, b, settings)
                    if failed_gate is None:
                        score = _combined_score(scores, settings)
                        cost[r, c] = -score
                        feasible[r, c] = True
                        pair_scores[(r, c)] = {**scores, "combined_score": score}
                    else:
                        rejected.append(_entry(a, b, failed_gate, scores))

            if not feasible.any():
                continue

            row_idx, col_idx = linear_sum_assignment(cost)
            chosen = set()
            for r, c in zip(row_idx, col_idx):
                # Hungarian fills the matrix; drop assignments that landed on
                # cells no gate ever admitted.
                if not feasible[r, c]:
                    continue
                a, b = rows[r], cols[c]
                uf.union(a.key, b.key)
                has_successor.add(a.key)
                has_predecessor.add(b.key)
                accepted_scores[b.key] = pair_scores[(r, c)]
                chosen.add((r, c))
                n_links += 1

            for (r, c), scores in pair_scores.items():
                if (r, c) not in chosen:
                    suppressed.append(_entry(rows[r], cols[c], "assignment_conflict", scores))

    # --- Materialize global objects ----------------------------------------
    all_tracks = [t for clip in clip_summaries for t in clip]
    by_key = {t.key: t for t in all_tracks}

    components: dict[str, list[_ClipTrackSummary]] = {}
    for track in all_tracks:
        components.setdefault(uf.find(track.key), []).append(track)

    ordered_roots = sorted(components, key=lambda root: min(t.sort_key for t in components[root]))

    objects: list[GlobalObject] = []
    for n, root in enumerate(ordered_roots, start=1):
        members = sorted(components[root], key=lambda t: t.sort_key)
        objects.append(
            GlobalObject(
                global_id=f"obj_{n:04d}",
                **{"class": members[0].cls},
                sightings=[
                    Sighting(
                        clip_id=m.clip_id,
                        track_id=m.track_id,
                        first_seen=m.first_wallclock,
                        last_seen=m.last_wallclock,
                        # Scores of the link that joined this sighting to the
                        # previous one; empty for the first sighting.
                        gate_scores=accepted_scores.get(m.key, {}),
                    )
                    for m in members
                ],
            )
        )

    index = MasterObjectIndex(
        video_id=video_id,
        objects=objects,
        rejected_links=rejected,
        suppressed_links=suppressed,
    )

    output_dir = outputs_dir / "master_object_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(index.model_dump_json(indent=2, by_alias=True))

    n_multi_clip = sum(1 for o in objects if len(o.sightings) > 1)
    logger.info(
        "link_video: video_id=%s %d tracks -> %d global objects (%d spanning multiple clips, "
        "%d links, %d gated-out, %d suppressed) -> %s",
        video_id,
        n_tracks,
        len(objects),
        n_multi_clip,
        n_links,
        len(rejected),
        len(suppressed),
        output_path,
    )

    return index
