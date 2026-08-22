"""M12 — Hybrid retrieval: question -> RetrievalResult.

    1. QUERY UNDERSTANDING  question -> QueryIntent            (src/retrieval/intent.py)
    2. VECTOR RETRIEVAL     semantic search + metadata filters (Chroma, M11)
    3. GRAPH RETRIEVAL      templated parameterized Cypher     (Neo4j, M10)
    4. FUSION               merge, dedupe, rank
    5. CONTEXT ASSEMBLY     full timelines/attributes/events/evidence for top-K

FUSION. The two retrievers do not return the same kind of signal, so their
scores are not blended directly. Vector search gives *graded, fuzzy*
relevance; the graph gives *binary, exact* constraint satisfaction. Min-max
normalizing both onto [0,1] would assert an exchange rate between a cosine
distance and "the KG confirms this is a white sedan" that does not exist. So:

    score = w_vec  * RRF(vector_rank)              graded relevance
          + w_graph * 1[satisfies the constraints]  exact, authoritative
          + w_conf  * mean_confidence(matched attrs) the system's own certainty

Reciprocal Rank Fusion is rank-based and therefore needs no calibration
between systems -- which is precisely the incommensurability above. It is
normalized so rank 1 scores 1.0, putting all three terms on a comparable
scale and making the weights directly interpretable.

The graph contributes *membership*, not a rank. A template like "all cars"
returns an unordered set; assigning it ranks 1..N would fabricate ordering
information that does not exist.

The confidence term is what makes retrieval respect the pipeline's own
uncertainty: an object M8 confirmed white at 1.0 outranks one where "white"
won a coin-flip M6 flagged uncertain. It contributes only when the question
actually constrained on attributes, so it stays neutral otherwise.

TEMPLATED CYPHER, NEVER GENERATED. Query structure is code; only values cross
the boundary, as bound parameters. Note that Cypher cannot parameterize
relationship types, so event filtering goes through the `Event.type` property
rather than a `-[:OVERTAKES]->` type -- which works because M10 kept full
Event nodes alongside the denormalized shortcut edges.

Neo4j being unreachable degrades to vector-only with a warning on the result,
rather than failing the question outright.
"""
from __future__ import annotations

from typing import Any

from src.retrieval.intent import parse_intent
from src.utils.config import PipelineSettings, get_settings
from src.utils.logging import get_logger
from src.utils.schemas import QueryIntent, RetrievalResult, RetrievedObject

logger = get_logger(__name__)

# Attributes that may appear in a WHERE clause, mapped to their node property.
# An allowlist, so a parsed intent can never name an arbitrary property.
_FILTERABLE_ATTRIBUTES = {
    "color": "color",
    "vehicle_type": "vehicle_type",
    "make": "make",
    "direction": "direction",
}

# Pairwise event type -> the M10 shortcut relationship that encodes it.
# Code-defined, so the literal spliced into Cypher can never come from input.
_PAIRWISE_RELATIONSHIP = {"OVERTAKE": "OVERTAKES"}


class HybridRetrieverError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 2. Vector retrieval
# ---------------------------------------------------------------------------


def build_vector_filter(video_id: str, intent: QueryIntent) -> dict:
    """Chroma `where` clause from the structured intent.

    Only equality on scalar metadata, matching how M11 stored it -- event
    membership is a boolean column per type precisely so this stays a plain
    equality filter.
    """
    conditions: list[dict] = [{"video_id": video_id}]

    for name, value in sorted(intent.target_attributes.items()):
        if name in _FILTERABLE_ATTRIBUTES:
            conditions.append({name: value})

    # A single class is a filter; several are a disjunction Chroma cannot
    # express alongside other ANDs cleanly, so those are left to the graph
    # and to semantic similarity instead of over-constraining here.
    if len(intent.object_classes) == 1:
        conditions.append({"class": intent.object_classes[0]})

    for event_type in sorted(intent.event_types):
        conditions.append({f"event_{event_type}": True})

    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def vector_search(
    video_id: str, intent: QueryIntent, settings: PipelineSettings
) -> list[dict]:
    from src.retrieval.vector_store import get_chroma_client

    client = get_chroma_client(settings)
    try:
        collection = client.get_collection(settings.vector_store.object_collection)
    except Exception as exc:  # collection absent -> nothing indexed yet
        raise HybridRetrieverError(
            f"No object collection in the vector store; run `index {video_id}` first."
        ) from exc

    result = collection.query(
        query_texts=[intent.question],
        n_results=settings.retrieval.vector_candidates,
        where=build_vector_filter(video_id, intent),
    )

    hits = []
    for rank, (doc_id, document, distance, metadata) in enumerate(
        zip(result["ids"][0], result["documents"][0], result["distances"][0], result["metadatas"][0]),
        start=1,
    ):
        hits.append(
            {
                "global_id": doc_id,
                "rank": rank,
                "distance": distance,
                "summary": document,
                "metadata": metadata,
            }
        )
    return hits


# ---------------------------------------------------------------------------
# 3. Graph retrieval — templated Cypher
# ---------------------------------------------------------------------------


def build_object_query(video_id: str, intent: QueryIntent, limit: int) -> tuple[str, dict]:
    """Assemble a parameterized object query from a fixed predicate set.

    Every predicate below is written in code; the intent only supplies bound
    values. Returns (cypher, params).
    """
    predicates: list[str] = []
    params: dict[str, Any] = {"video_id": video_id, "limit": limit}

    for name, value in sorted(intent.target_attributes.items()):
        prop = _FILTERABLE_ATTRIBUTES.get(name)
        if prop:
            predicates.append(f"o.{prop} = $attr_{name}")
            params[f"attr_{name}"] = value

    if intent.object_classes:
        predicates.append("o.class IN $classes")
        params["classes"] = sorted(intent.object_classes)

    # Subject-role only, matching M11's metadata semantics: "vehicles that
    # overtook" must not also return the vehicle that was overtaken.
    for i, event_type in enumerate(sorted(intent.event_types)):
        predicates.append(
            f"EXISTS {{ MATCH (ev{i}:Event {{video_id: $video_id, type: $event_type_{i}}})"
            f"-[:INVOLVES {{role: 'subject'}}]->(o) }}"
        )
        params[f"event_type_{i}"] = event_type

    # Relational constraint: the counterpart of a pairwise event. The
    # relationship type is a literal chosen from a code-defined map (Cypher
    # cannot parameterize types), while its properties stay bound parameters.
    if intent.relations and (intent.counterpart_classes or intent.counterpart_attributes):
        rel = _PAIRWISE_RELATIONSHIP.get(intent.relations[0])
        if rel:
            counterpart_predicates = []
            if intent.counterpart_classes:
                counterpart_predicates.append("other.class IN $counterpart_classes")
                params["counterpart_classes"] = sorted(intent.counterpart_classes)
            for name, value in sorted(intent.counterpart_attributes.items()):
                prop = _FILTERABLE_ATTRIBUTES.get(name)
                if prop:
                    counterpart_predicates.append(f"other.{prop} = $counterpart_{name}")
                    params[f"counterpart_{name}"] = value
            if counterpart_predicates:
                predicates.append(
                    f"EXISTS {{ MATCH (o)-[:{rel}]->(other:TrafficObject) "
                    f"WHERE {' AND '.join(counterpart_predicates)} }}"
                )

    for i, region_id in enumerate(sorted(intent.region_ids)):
        predicates.append(
            f"EXISTS {{ MATCH (o)-[:CROSSES]->(:Location {{region_id: $region_{i}}}) }}"
        )
        params[f"region_{i}"] = region_id

    # Time-of-day slice of the ISO timestamp: "2026-08-15T12:05:00" -> "12:05:00".
    if intent.time_range.after:
        predicates.append("substring(o.last_seen, 11, 8) >= $after")
        params["after"] = intent.time_range.after
    if intent.time_range.before:
        predicates.append("substring(o.first_seen, 11, 8) <= $before")
        params["before"] = intent.time_range.before

    where = ("WHERE " + " AND ".join(predicates)) if predicates else ""
    cypher = f"""
MATCH (o:TrafficObject {{video_id: $video_id}})
{where}
RETURN o.global_id AS global_id, o.class AS class, o.color AS color,
       o.vehicle_type AS vehicle_type, o.make AS make,
       o.first_seen AS first_seen, o.last_seen AS last_seen,
       o.color_confidence AS color_confidence,
       o.vehicle_type_confidence AS vehicle_type_confidence,
       o.make_confidence AS make_confidence,
       o.direction_confidence AS direction_confidence
ORDER BY o.first_seen, o.global_id
LIMIT $limit
""".strip()
    return cypher, params


def build_count_query(video_id: str, intent: QueryIntent) -> tuple[str, dict]:
    """Counting asks a different question than ranking, so it gets its own
    template: a grouped count over the same predicates, not a top-K list."""
    object_query, params = build_object_query(video_id, intent, limit=0)
    body = object_query.split("RETURN")[0].strip()
    params.pop("limit", None)
    cypher = f"""
{body}
RETURN o.class AS class, coalesce(o.color, 'unknown') AS color,
       coalesce(o.vehicle_type, 'unknown') AS vehicle_type, count(*) AS n
ORDER BY n DESC, class, color
""".strip()
    return cypher, params


def build_event_query(video_id: str, intent: QueryIntent, limit: int) -> tuple[str, dict]:
    predicates = ["e.video_id = $video_id"]
    params: dict[str, Any] = {"video_id": video_id, "limit": limit}
    if intent.event_types:
        predicates.append("e.type IN $event_types")
        params["event_types"] = sorted(intent.event_types)
    if intent.time_range.after:
        predicates.append("substring(e.start_time, 11, 8) >= $after")
        params["after"] = intent.time_range.after
    if intent.time_range.before:
        predicates.append("substring(e.start_time, 11, 8) <= $before")
        params["before"] = intent.time_range.before

    cypher = f"""
MATCH (e:Event)-[r:INVOLVES]->(o:TrafficObject)
WHERE {" AND ".join(predicates)}
RETURN e.event_id AS event_id, e.type AS type, e.start_time AS start_time,
       e.end_time AS end_time, e.confidence AS confidence,
       e.evidence_frames AS evidence_frames,
       collect({{global_id: o.global_id, role: r.role}}) AS participants
ORDER BY e.start_time, e.event_id
LIMIT $limit
""".strip()
    return cypher, params


CONTEXT_QUERY = """
MATCH (o:TrafficObject {video_id: $video_id})
WHERE o.global_id IN $global_ids
OPTIONAL MATCH (o)-[ha:HAS_ATTRIBUTE]->(av:AttributeValue)
WITH o, collect(DISTINCT {type: av.type, value: av.value, share: ha.share,
                          winner: ha.winner, confidence: ha.confidence,
                          source: ha.source, uncertain: ha.uncertain}) AS attributes
OPTIONAL MATCH (o)-[:SEEN_IN]->(c:Clip)
WITH o, attributes, collect(DISTINCT c.clip_id) AS clips
OPTIONAL MATCH (ev:Event)-[role:INVOLVES]->(o)
WITH o, attributes, clips,
     collect(DISTINCT {event_id: ev.event_id, type: ev.type, role: role.role,
                       start_time: ev.start_time, end_time: ev.end_time,
                       confidence: ev.confidence,
                       evidence_frames: ev.evidence_frames}) AS events
OPTIONAL MATCH (o)-[:SEEN_AT]->(f:Frame)
RETURN o.global_id AS global_id, o.class AS class, attributes, clips, events,
       collect(DISTINCT f.frame_path)[0..$max_frames] AS evidence_frames,
       collect(DISTINCT f.wallclock_time) AS sighting_times
""".strip()


# ---------------------------------------------------------------------------
# 4. Fusion
# ---------------------------------------------------------------------------


def rrf_score(rank: int, k: int) -> float:
    """Reciprocal Rank Fusion, normalized so rank 1 scores exactly 1.0.

    Normalizing matters: raw RRF values sit near 1/k (~0.016 at k=60), which
    would make the weights meaningless against the other two [0,1] terms.
    """
    return (k + 1) / (k + rank)


def _matched_confidence(intent: QueryIntent, graph_row: dict | None, metadata: dict | None) -> float:
    """Mean confidence of the attributes the question actually constrained.

    Neutral (0.0) when the question named no attributes -- there is nothing to
    be confident *about*, and a blanket bonus would just add noise.
    """
    if not intent.target_attributes:
        return 0.0
    confidences = []
    for name in intent.target_attributes:
        value = None
        if graph_row is not None:
            value = graph_row.get(f"{name}_confidence")
        if value is None and metadata is not None:
            value = metadata.get(f"{name}_confidence")
        if value is not None:
            confidences.append(float(value))
    return sum(confidences) / len(confidences) if confidences else 0.0


def fuse(
    intent: QueryIntent,
    vector_hits: list[dict],
    graph_rows: list[dict],
    settings: PipelineSettings,
    graph_available: bool = True,
) -> list[RetrievedObject]:
    weights = settings.retrieval_weights
    k = settings.retrieval.rrf_k

    graph_by_id = {row["global_id"]: row for row in graph_rows}
    vector_by_id = {hit["global_id"]: hit for hit in vector_hits}

    # When the question carries hard constraints and the graph could answer,
    # graph membership is a FILTER, not merely a bonus: an object the KG did
    # not return has failed an explicit constraint, and surfacing it anyway
    # would answer "before 11:04" with something first seen at 11:06. Vector
    # search cannot enforce these itself (Chroma has no range predicate over
    # our ISO timestamps), so the graph is the only place they can hold.
    # Without constraints, or with the graph down, fall back to the union.
    if graph_available and intent.has_structured_constraints():
        candidate_ids = set(graph_by_id)
    else:
        candidate_ids = set(graph_by_id) | set(vector_by_id)

    fused: list[RetrievedObject] = []
    for global_id in sorted(candidate_ids):
        hit = vector_by_id.get(global_id)
        row = graph_by_id.get(global_id)
        metadata = hit.get("metadata") if hit else None

        vector_term = rrf_score(hit["rank"], k) if hit else 0.0
        graph_term = 1.0 if row is not None else 0.0
        confidence_term = _matched_confidence(intent, row, metadata)

        sources = [name for name, present in (("vector", hit), ("graph", row)) if present]
        fused.append(
            RetrievedObject(
                global_id=global_id,
                **{"class": (row or {}).get("class") or (metadata or {}).get("class") or "unknown"},
                score=round(
                    weights.vector * vector_term
                    + weights.graph * graph_term
                    + weights.confidence * confidence_term,
                    6,
                ),
                sources=sources,
                vector_rank=hit["rank"] if hit else None,
                vector_distance=hit["distance"] if hit else None,
                graph_matched=row is not None,
                matched_confidence=round(confidence_term, 6),
                timeline_summary=hit["summary"] if hit else "",
            )
        )

    # Ties broken by id so the ordering is reproducible run to run.
    fused.sort(key=lambda o: (-o.score, o.global_id))
    return fused


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_graph(
    video_id: str, intent: QueryIntent, settings: PipelineSettings, result: RetrievalResult
) -> tuple[list[dict], list[dict]]:
    """Graph half. On a connection failure, warns and returns empties so the
    question is still answered from the vector store alone."""
    from neo4j import GraphDatabase

    cfg = settings.neo4j
    try:
        driver = GraphDatabase.driver(
            cfg.uri, auth=(cfg.user, cfg.password), notifications_min_severity="WARNING"
        )
        driver.verify_connectivity()
    except Exception as exc:
        result.warnings.append(
            f"Neo4j unreachable ({exc.__class__.__name__}); answered from the vector store only, "
            "so exact constraints were not enforced."
        )
        logger.warning("hybrid_retrieve: Neo4j unreachable, degrading to vector-only")
        return [], []

    try:
        with driver.session(database=cfg.database) as session:
            if intent.question_type == "counting":
                cypher, params = build_count_query(video_id, intent)
                result.cypher_queries.append(cypher)
                breakdown = [dict(record) for record in session.run(cypher, **params)]
                result.count_breakdown = breakdown
                result.count = sum(row["n"] for row in breakdown)

            cypher, params = build_object_query(video_id, intent, settings.retrieval.graph_limit)
            result.cypher_queries.append(cypher)
            object_rows = [dict(record) for record in session.run(cypher, **params)]

            event_rows: list[dict] = []
            if intent.event_types or intent.question_type != "factual":
                cypher, params = build_event_query(video_id, intent, settings.retrieval.graph_limit)
                result.cypher_queries.append(cypher)
                event_rows = [dict(record) for record in session.run(cypher, **params)]

            return object_rows, event_rows
    finally:
        driver.close()


def _assemble_context(
    video_id: str, global_ids: list[str], settings: PipelineSettings
) -> dict[str, dict]:
    if not global_ids:
        return {}
    from neo4j import GraphDatabase

    cfg = settings.neo4j
    driver = GraphDatabase.driver(
        cfg.uri, auth=(cfg.user, cfg.password), notifications_min_severity="WARNING"
    )
    try:
        with driver.session(database=cfg.database) as session:
            records = session.run(
                CONTEXT_QUERY, video_id=video_id, global_ids=global_ids, max_frames=5
            )
            return {record["global_id"]: dict(record) for record in records}
    finally:
        driver.close()


def hybrid_retrieve(
    question: str, video_id: str, settings: PipelineSettings | None = None
) -> RetrievalResult:
    settings = settings or get_settings()

    intent = parse_intent(question, settings)
    result = RetrievalResult(question=question, video_id=video_id, intent=intent)

    top_k = settings.retrieval.top_k
    if intent.question_type in ("counterfactual", "forecast"):
        top_k *= settings.retrieval.reasoning_widen_factor

    # --- vector half ---
    vector_hits: list[dict] = []
    try:
        vector_hits = vector_search(video_id, intent, settings)
    except HybridRetrieverError as exc:
        result.warnings.append(str(exc))
    except Exception as exc:
        result.warnings.append(f"Vector search failed ({exc.__class__.__name__}); graph results only.")

    # --- graph half ---
    graph_rows, event_rows = _run_graph(video_id, intent, settings, result)
    result.events = event_rows
    graph_available = not any("Neo4j unreachable" in w for w in result.warnings)

    # Counting is answered by the graph outright; a similarity ranking cannot
    # count, so the object list is context for the answer, not the answer.
    fused = fuse(intent, vector_hits, graph_rows, settings, graph_available=graph_available)
    top = fused[:top_k]

    if top and not any("Neo4j unreachable" in w for w in result.warnings):
        context = _assemble_context(video_id, [o.global_id for o in top], settings)
        for obj in top:
            record = context.get(obj.global_id)
            if not record:
                continue
            obj.attributes = [a for a in record.get("attributes") or [] if a.get("type")]
            obj.clips = sorted(c for c in record.get("clips") or [] if c)
            obj.events = [e for e in record.get("events") or [] if e.get("event_id")]
            obj.evidence_frames = [f for f in record.get("evidence_frames") or [] if f]
            # Sorted so the earliest sighting is first: the context offers one
            # citable time per object, and "when it was first seen" is the
            # least arbitrary choice.
            obj.sighting_times = sorted(t for t in record.get("sighting_times") or [] if t)
            if obj.cls == "unknown" and record.get("class"):
                obj.cls = record["class"]

    result.objects = top

    logger.info(
        "hybrid_retrieve: video_id=%s type=%s vector=%d graph=%d fused=%d returned=%d%s",
        video_id,
        intent.question_type,
        len(vector_hits),
        len(graph_rows),
        len(fused),
        len(top),
        f" count={result.count}" if result.count is not None else "",
    )
    return result
