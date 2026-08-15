"""M10 — Knowledge graph construction (Neo4j).

Materializes the pipeline's JSON outputs into a queryable graph. See
src/graph/schema.cypher for the schema and the reasoning behind storing
attributes both denormalized (indexed winner) and normalized (full evidence).

STRUCTURE. Reading/shaping is separated from writing: `collect_payload()` is a
pure function from JSON files to a `GraphPayload` of plain dicts, and
`write_payload()` is the only part that touches Neo4j. That split is what lets
the graph's contents be tested without a live database.

IDEMPOTENCY. Re-running must not duplicate, and must also not leave behind
objects a re-run no longer produces -- MERGE alone would do the first but not
the second. So a load is a scoped delete-then-insert: every node carrying this
video_id is detached and deleted, then rebuilt, all inside one transaction, so
a failure rolls back rather than leaving a half-graph. AttributeValue nodes are
shared vocabulary across videos and are deliberately never deleted.

Inputs (all optional except the first two -- missing ones just narrow the graph):
    master_object_index/<video_id>.json  or global_objects_final/<video_id>.json
    ingest/<video_id>_manifest.json
    tracks_associated/<clip_id>.json      -> per-frame sightings
    attributes_canonical/<clip_id>.json   -> vote distributions
    events/<video_id>.json                -> events + typed shortcut edges
    configs/regions/<video_id>.yaml       -> Location nodes
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.graph.event_detector import load_regions
from src.semantics.vocabulary import ATTRIBUTES
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import (
    ClipAssociatedTracks,
    ClipCanonicalAttributes,
    EventLog,
    FinalObjectIndex,
    MasterObjectIndex,
    VideoManifest,
)

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.cypher")

# Event types that also get a denormalized shortcut edge, so the common
# question ("who overtook whom?") is one hop rather than a two-hop join
# through the Event node. Same materialized-view argument as the attributes.
_SHORTCUT_EDGES = {
    "OVERTAKE": "OVERTAKES",
    "STOP": "STOPS_AT",
    "PARK": "PARKED_AT",
    "CROSSES": "CROSSES",
}


class KGBuilderError(RuntimeError):
    pass


@dataclass
class GraphPayload:
    """Everything to be written, as plain dicts ready for UNWIND."""

    video: dict[str, Any] = field(default_factory=dict)
    clips: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)
    attribute_edges: list[dict] = field(default_factory=list)
    clip_sightings: list[dict] = field(default_factory=list)
    frame_sightings: list[dict] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    event_roles: list[dict] = field(default_factory=list)
    shortcut_edges: list[dict] = field(default_factory=list)
    moved_to: list[dict] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "clips": len(self.clips),
            "frames": len(self.frames),
            "objects": len(self.objects),
            "attribute_edges": len(self.attribute_edges),
            "frame_sightings": len(self.frame_sightings),
            "locations": len(self.locations),
            "events": len(self.events),
            "shortcut_edges": len(self.shortcut_edges),
        }


def _uid(video_id: str, local_id: str) -> str:
    """Video-scoped key. `obj_0001` recurs in every video, so raw ids cannot
    carry a uniqueness constraint."""
    return f"{video_id}:{local_id}"


def _aggregate_distributions(
    canonical_by_clip: dict[str, dict[str, dict[str, dict]]],
    sightings: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Merge M6's per-clip vote distributions for one global object.

    Each sighting's distribution is weighted by how many observations backed
    it, so a sighting seen across 20 frames outweighs one seen across 3.
    Uncertain sightings still contribute their distribution here -- unlike the
    M7 gate, this is evidence display, not a decision, and hiding the shape of
    a close call would defeat the purpose of storing it.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for attribute in ATTRIBUTES:
        totals: dict[str, float] = defaultdict(float)
        for clip_id, track_id in sightings:
            vote = canonical_by_clip.get(clip_id, {}).get(track_id, {}).get(attribute)
            if not vote:
                continue
            weight = max(vote.get("n_votes", 0), 1)
            for value, share in (vote.get("distribution") or {}).items():
                totals[value] += weight * share
        total = sum(totals.values())
        if total > 0:
            aggregated[attribute] = {v: w / total for v, w in totals.items()}
    return aggregated


def collect_payload(video_id: str, settings: PipelineSettings | None = None) -> GraphPayload:
    """Read every pipeline output for `video_id` and shape it for loading.

    Pure with respect to Neo4j: no connection is opened, which is what makes
    the graph's contents testable without a database.
    """
    settings = settings or get_settings()
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    payload = GraphPayload()

    # --- Video / Clip / Frame ------------------------------------------------
    manifest_path = outputs / "ingest" / f"{video_id}_manifest.json"
    if not manifest_path.exists():
        raise KGBuilderError(f"No ingest manifest at {manifest_path}; run `ingest` first.")
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())

    payload.video = {
        "video_id": video_id,
        "source_path": manifest.source_path,
        "fps": manifest.fps,
        "width": manifest.resolution[0],
        "height": manifest.resolution[1],
        "frame_count": manifest.frame_count,
        "start_wallclock": manifest.start_wallclock,
        "end_wallclock": manifest.end_wallclock,
    }

    for clip in manifest.clips:
        payload.clips.append(
            {
                "uid": _uid(video_id, clip.clip_id),
                "video_id": video_id,
                "clip_id": clip.clip_id,
                "clip_path": clip.clip_path,
                "start_wallclock": clip.start_wallclock,
                "end_wallclock": clip.end_wallclock,
                "fps": clip.fps,
            }
        )
        for frame in clip.frames:
            payload.frames.append(
                {
                    "uid": _uid(video_id, frame.frame_id),
                    "video_id": video_id,
                    "clip_uid": _uid(video_id, clip.clip_id),
                    "frame_id": frame.frame_id,
                    "frame_path": frame.frame_path,
                    "video_timestamp_sec": frame.video_timestamp_sec,
                    "wallclock_time": frame.wallclock_time,
                }
            )

    # --- Objects: prefer M8's confirmed attributes, fall back to M7 ----------
    final_path = outputs / "global_objects_final" / f"{video_id}.json"
    master_path = outputs / "master_object_index" / f"{video_id}.json"

    final_index: FinalObjectIndex | None = None
    if final_path.exists():
        final_index = FinalObjectIndex.model_validate_json(final_path.read_text())
        object_records = [
            (o.global_id, o.cls, [(s.clip_id, s.track_id, s.first_seen, s.last_seen) for s in o.sightings],
             {a.attribute: a for a in o.attributes})
            for o in final_index.objects
        ]
    elif master_path.exists():
        master = MasterObjectIndex.model_validate_json(master_path.read_text())
        logger.warning(
            "collect_payload: no confirmed attributes for %s; loading M7 identities "
            "without attributes (run `confirm` for the full graph)",
            video_id,
        )
        object_records = [
            (o.global_id, o.cls, [(s.clip_id, s.track_id, s.first_seen, s.last_seen) for s in o.sightings], {})
            for o in master.objects
        ]
    else:
        raise KGBuilderError(
            f"No object index for {video_id} (looked in {final_path} and {master_path}); run `link` first."
        )

    # --- Per-clip joins: tracks (positions) and canonical votes (evidence) ---
    clip_ids = sorted({clip_id for _, _, sightings, _ in object_records for clip_id, _, _, _ in sightings})
    tracks_by_clip: dict[str, dict[str, Any]] = {}
    canonical_by_clip: dict[str, dict[str, dict[str, dict]]] = {}

    for clip_id in clip_ids:
        tracks_path = outputs / "tracks_associated" / f"{clip_id}.json"
        if tracks_path.exists():
            clip_tracks = ClipAssociatedTracks.model_validate_json(tracks_path.read_text())
            tracks_by_clip[clip_id] = {t.track_id: t for t in clip_tracks.tracks}

        canonical_path = outputs / "attributes_canonical" / f"{clip_id}.json"
        if canonical_path.exists():
            canonical = ClipCanonicalAttributes.model_validate_json(canonical_path.read_text())
            canonical_by_clip[clip_id] = {
                t.track_id: {
                    v.attribute: {"distribution": v.distribution, "n_votes": v.n_votes}
                    for v in t.votes
                }
                for t in canonical.tracks
            }

    frame_uid_by_id = {f["frame_id"]: f["uid"] for f in payload.frames}

    for global_id, cls, sightings, final_attributes in object_records:
        object_uid = _uid(video_id, global_id)
        node = {
            "uid": object_uid,
            "video_id": video_id,
            "global_id": global_id,
            "class": cls,
            "label": "Person" if cls == "person" else "Vehicle",
            "n_sightings": len(sightings),
            "first_seen": min((s[2] for s in sightings), default=None),
            "last_seen": max((s[3] for s in sightings), default=None),
        }
        # Denormalize the winning value + its provenance for indexed filtering.
        for attribute in ATTRIBUTES:
            final = final_attributes.get(attribute)
            node[attribute] = final.value if final else None
            node[f"{attribute}_confidence"] = final.confidence if final else 0.0
            node[f"{attribute}_source"] = final.source if final else "unconfirmed"
            node[f"{attribute}_uncertain"] = final.uncertain if final else True
        payload.objects.append(node)

        # Normalize the evidence: one edge per candidate value.
        distributions = _aggregate_distributions(
            canonical_by_clip, [(clip_id, track_id) for clip_id, track_id, _, _ in sightings]
        )
        for attribute in ATTRIBUTES:
            final = final_attributes.get(attribute)
            winner = final.value if final else None
            candidates = dict(distributions.get(attribute, {}))
            # M8 can answer where clip-level voting never did, so the winner
            # may not appear in any clip-level distribution.
            if winner and winner not in candidates:
                candidates[winner] = 1.0 if not candidates else 0.0
            for value, share in sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0])):
                payload.attribute_edges.append(
                    {
                        "object_uid": object_uid,
                        "key": f"{attribute}:{value}",
                        "type": attribute,
                        "value": value,
                        "share": round(share, 6),
                        "winner": value == winner,
                        "confidence": (final.confidence if final and value == winner else 0.0),
                        "source": (final.source if final and value == winner else "clip_voting"),
                        "uncertain": (final.uncertain if final and value == winner else True),
                    }
                )

        for clip_id, track_id, first_seen, last_seen in sightings:
            payload.clip_sightings.append(
                {
                    "object_uid": object_uid,
                    "clip_uid": _uid(video_id, clip_id),
                    "track_id": track_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                }
            )
            track = tracks_by_clip.get(clip_id, {}).get(track_id)
            if track is None:
                continue
            for frame_id, bbox in zip(track.frames, track.bboxes):
                frame_uid = frame_uid_by_id.get(frame_id)
                if frame_uid is None:
                    continue
                payload.frame_sightings.append(
                    {
                        "object_uid": object_uid,
                        "frame_uid": frame_uid,
                        "bbox": [float(v) for v in bbox],
                    }
                )

    # --- Locations from the per-video regions file ---------------------------
    lines, polygons = load_regions(video_id, settings)
    for region, region_type in [(l, "line") for l in lines] + [(p, "polygon") for p in polygons]:
        payload.locations.append(
            {
                "uid": _uid(video_id, region.id),
                "video_id": video_id,
                "region_id": region.id,
                "region_type": region_type,
            }
        )

    # --- Events + typed shortcut edges ---------------------------------------
    events_path = outputs / "events" / f"{video_id}.json"
    if events_path.exists():
        event_log = EventLog.model_validate_json(events_path.read_text())
        for event in event_log.events:
            event_uid = _uid(video_id, event.event_id)
            payload.events.append(
                {
                    "uid": event_uid,
                    "video_id": video_id,
                    "event_id": event.event_id,
                    "type": event.type,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "confidence": event.confidence,
                    "evidence_frames": event.evidence_frames,
                    **{f"meta_{k}": v for k, v in event.metadata.items()},
                }
            )
            payload.event_roles.append(
                {"event_uid": event_uid, "object_uid": _uid(video_id, event.subject_id), "role": "subject"}
            )
            if event.object_id:
                payload.event_roles.append(
                    {"event_uid": event_uid, "object_uid": _uid(video_id, event.object_id), "role": "object"}
                )

            rel = _SHORTCUT_EDGES.get(event.type)
            if rel is None:
                continue

            if event.type == "OVERTAKE" and event.object_id:
                payload.shortcut_edges.append(
                    {
                        "rel": rel, "from_uid": _uid(video_id, event.subject_id),
                        "to_uid": _uid(video_id, event.object_id), "to_kind": "object",
                        "event_id": event.event_id, "at": event.start_time,
                        "end": event.end_time, "confidence": event.confidence,
                    }
                )
            elif event.type == "CROSSES":
                region_id = event.metadata.get("region_id")
                if region_id:
                    payload.shortcut_edges.append(
                        {
                            "rel": rel, "from_uid": _uid(video_id, event.subject_id),
                            "to_uid": _uid(video_id, region_id), "to_kind": "location",
                            "event_id": event.event_id, "at": event.start_time,
                            "end": event.end_time, "confidence": event.confidence,
                        }
                    )
            elif event.evidence_frames:
                frame_uid = frame_uid_by_id.get(event.evidence_frames[0])
                if frame_uid:
                    payload.shortcut_edges.append(
                        {
                            "rel": rel, "from_uid": _uid(video_id, event.subject_id),
                            "to_uid": frame_uid, "to_kind": "frame",
                            "event_id": event.event_id, "at": event.start_time,
                            "end": event.end_time, "confidence": event.confidence,
                        }
                    )

        # MOVED_TO: the ordered sequence of regions an object actually entered,
        # derived from its CROSSES events. Empty when no regions are configured.
        crossings: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for event in event_log.events:
            if event.type == "CROSSES" and event.metadata.get("region_id"):
                crossings[event.subject_id].append((event.start_time, event.metadata["region_id"]))
        for subject_id, entries in crossings.items():
            seen: list[str] = []
            for _, region_id in sorted(entries):
                if not seen or seen[-1] != region_id:
                    seen.append(region_id)
            for order, region_id in enumerate(seen, start=1):
                payload.moved_to.append(
                    {
                        "object_uid": _uid(video_id, subject_id),
                        "location_uid": _uid(video_id, region_id),
                        "order": order,
                    }
                )

    return payload


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def split_cypher_statements(text: str) -> list[str]:
    """Split a .cypher file into executable statements.

    Comments are stripped BEFORE splitting on ';'. Doing it the other way round
    means a semicolon inside a comment tears a statement in half and leaves the
    remainder of the comment as a syntax error.
    """
    body = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("//"))
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


def _schema_statements() -> list[str]:
    return split_cypher_statements(SCHEMA_PATH.read_text())


# Node labels that carry a video_id and are therefore rebuilt on every load.
# AttributeValue is intentionally absent: it is shared vocabulary across videos.
_VIDEO_SCOPED_LABELS = ("Frame", "Clip", "Event", "Location", "TrafficObject", "Video")


def _delete_video(tx, video_id: str) -> None:
    for label in _VIDEO_SCOPED_LABELS:
        tx.run(f"MATCH (n:{label} {{video_id: $video_id}}) DETACH DELETE n", video_id=video_id)


def _write_payload(tx, payload: GraphPayload, batch_size: int) -> None:
    def batches(rows: list[dict]):
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    tx.run(
        """
        MERGE (v:Video {video_id: $video_id})
        SET v += $props
        """,
        video_id=payload.video["video_id"],
        props=payload.video,
    )

    for rows in batches(payload.clips):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (c:Clip {uid: row.uid})
            SET c += row
            WITH c, row
            MATCH (v:Video {video_id: row.video_id})
            MERGE (c)-[:PART_OF]->(v)
            """,
            rows=rows,
        )

    for rows in batches(payload.frames):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (f:Frame {uid: row.uid})
            SET f += row
            WITH f, row
            MATCH (c:Clip {uid: row.clip_uid})
            MERGE (f)-[:PART_OF]->(c)
            """,
            rows=rows,
        )

    # Vehicle/Person is a second label alongside TrafficObject, so queries can
    # target either the specific kind or anything trackable.
    for label in ("Vehicle", "Person"):
        rows_for_label = [o for o in payload.objects if o["label"] == label]
        for rows in batches(rows_for_label):
            tx.run(
                f"""
                UNWIND $rows AS row
                MERGE (o:TrafficObject {{uid: row.uid}})
                SET o += row, o:{label}
                """,
                rows=rows,
            )

    for rows in batches(payload.attribute_edges):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (o:TrafficObject {uid: row.object_uid})
            MERGE (a:AttributeValue {key: row.key})
            SET a.type = row.type, a.value = row.value
            MERGE (o)-[r:HAS_ATTRIBUTE {type: row.type, value: row.value}]->(a)
            SET r.share = row.share, r.winner = row.winner,
                r.confidence = row.confidence, r.source = row.source,
                r.uncertain = row.uncertain
            """,
            rows=rows,
        )

    for rows in batches(payload.clip_sightings):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (o:TrafficObject {uid: row.object_uid})
            MATCH (c:Clip {uid: row.clip_uid})
            MERGE (o)-[r:SEEN_IN {track_id: row.track_id}]->(c)
            SET r.first_seen = row.first_seen, r.last_seen = row.last_seen
            """,
            rows=rows,
        )

    for rows in batches(payload.frame_sightings):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (o:TrafficObject {uid: row.object_uid})
            MATCH (f:Frame {uid: row.frame_uid})
            MERGE (o)-[r:SEEN_AT]->(f)
            SET r.bbox = row.bbox
            """,
            rows=rows,
        )

    for rows in batches(payload.locations):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (l:Location {uid: row.uid})
            SET l += row
            """,
            rows=rows,
        )

    for rows in batches(payload.events):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (e:Event {uid: row.uid})
            SET e += row
            """,
            rows=rows,
        )

    for rows in batches(payload.event_roles):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (e:Event {uid: row.event_uid})
            MATCH (o:TrafficObject {uid: row.object_uid})
            MERGE (e)-[r:INVOLVES {role: row.role}]->(o)
            """,
            rows=rows,
        )

    # Relationship type cannot be parameterized in Cypher, so shortcut edges
    # are grouped by (rel, endpoint kind) and each group gets its own statement.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in payload.shortcut_edges:
        grouped[(edge["rel"], edge["to_kind"])].append(edge)

    target_label = {"object": "TrafficObject", "location": "Location", "frame": "Frame"}
    for (rel, to_kind), edges in sorted(grouped.items()):
        for rows in batches(edges):
            tx.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:TrafficObject {{uid: row.from_uid}})
                MATCH (b:{target_label[to_kind]} {{uid: row.to_uid}})
                MERGE (a)-[r:{rel} {{event_id: row.event_id}}]->(b)
                SET r.at = row.at, r.end = row.end, r.confidence = row.confidence
                """,
                rows=rows,
            )
            if rel == "OVERTAKES":
                # Generic pairwise edge alongside the specific one, so "did
                # these two ever interact?" does not need to enumerate types.
                tx.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:TrafficObject {uid: row.from_uid})
                    MATCH (b:TrafficObject {uid: row.to_uid})
                    MERGE (a)-[r:INTERACTS_WITH {event_id: row.event_id}]->(b)
                    SET r.type = 'OVERTAKE', r.at = row.at
                    """,
                    rows=rows,
                )

    for rows in batches(payload.moved_to):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (o:TrafficObject {uid: row.object_uid})
            MATCH (l:Location {uid: row.location_uid})
            MERGE (o)-[r:MOVED_TO {order: row.order}]->(l)
            """,
            rows=rows,
        )


def build_kg(video_id: str, settings: PipelineSettings | None = None) -> GraphPayload:
    settings = settings or get_settings()
    payload = collect_payload(video_id, settings)

    from neo4j import GraphDatabase

    cfg = settings.neo4j
    # Schema DDL is written with IF NOT EXISTS and re-run on every load, so
    # "already exists" notices are expected and would otherwise drown the log.
    driver = GraphDatabase.driver(
        cfg.uri, auth=(cfg.user, cfg.password), notifications_min_severity="WARNING"
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=cfg.database) as session:
            # Schema statements are DDL and cannot share a transaction with
            # the data writes.
            for statement in _schema_statements():
                session.run(statement)

            def _load(tx):
                _delete_video(tx, video_id)
                _write_payload(tx, payload, cfg.batch_size)

            session.execute_write(_load)
    finally:
        driver.close()

    logger.info(
        "build_kg: video_id=%s loaded %s",
        video_id,
        ", ".join(f"{k}={v}" for k, v in payload.counts().items()),
    )
    return payload
