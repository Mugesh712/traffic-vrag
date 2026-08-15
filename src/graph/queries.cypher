// M10 — Ten example queries covering the demo's question types.
// Each is standalone; $video_id and other $params are supplied by the caller.
// M12 turns these into parameterized templates rather than generating Cypher
// with an LLM, so the query surface stays fixed and safe.

// ---------------------------------------------------------------------------
// 1. FACTUAL / ATTRIBUTE FILTER — "show me all white sedans"
//    Hits the (color, vehicle_type) composite index directly. This is the
//    query the denormalized properties exist for.
// ---------------------------------------------------------------------------
MATCH (o:Vehicle {video_id: $video_id, color: 'white', vehicle_type: 'sedan'})
RETURN o.global_id AS object, o.color_confidence AS confidence,
       o.first_seen AS first_seen, o.last_seen AS last_seen
ORDER BY o.first_seen;

// ---------------------------------------------------------------------------
// 2. TEMPORAL — "which vehicles were present before 12:05?"
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id})-[:SEEN_AT]->(f:Frame)
WHERE f.wallclock_time < $before
RETURN o.global_id AS object, o.class AS class, o.color AS color,
       min(f.wallclock_time) AS first_seen, count(f) AS frames_seen
ORDER BY first_seen;

// ---------------------------------------------------------------------------
// 3. EVIDENCE / EXPLAINABILITY — full timeline of one object, with the frame
//    paths the UI needs to render proof.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id, global_id: $global_id})
OPTIONAL MATCH (o)-[:SEEN_IN]->(c:Clip)
OPTIONAL MATCH (e:Event)-[r:INVOLVES]->(o)
RETURN o.global_id AS object, o.class AS class,
       collect(DISTINCT {clip: c.clip_id, from: c.start_wallclock}) AS clips,
       collect(DISTINCT {event: e.type, role: r.role, at: e.start_time,
                         frames: e.evidence_frames}) AS events;

// ---------------------------------------------------------------------------
// 4. UNCERTAINTY — the vote distribution behind one attribute. This is what
//    the normalized HAS_ATTRIBUTE edges exist for; a serialized JSON property
//    could not answer it in Cypher.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id, global_id: $global_id})
      -[r:HAS_ATTRIBUTE {type: $attribute}]->(a:AttributeValue)
RETURN a.value AS value, r.share AS share, r.winner AS is_winner,
       r.confidence AS confidence, r.source AS decided_by, r.uncertain AS uncertain
ORDER BY r.share DESC;

// ---------------------------------------------------------------------------
// 5. INTERACTION — "who overtook whom?" One hop, thanks to the shortcut edge.
// ---------------------------------------------------------------------------
MATCH (a:TrafficObject {video_id: $video_id})-[r:OVERTAKES]->(b:TrafficObject)
RETURN a.global_id AS overtaker, a.color AS overtaker_color,
       b.global_id AS overtaken, b.color AS overtaken_color,
       r.at AS at, r.confidence AS confidence
ORDER BY r.at;

// ---------------------------------------------------------------------------
// 6. EVENT FILTER — vehicles that stopped, longest stop first.
// ---------------------------------------------------------------------------
MATCH (e:Event {video_id: $video_id})-[:INVOLVES {role: 'subject'}]->(o:TrafficObject)
WHERE e.type IN ['STOP', 'PARK']
RETURN o.global_id AS object, o.color AS color, o.vehicle_type AS type,
       e.type AS event, e.start_time AS from, e.end_time AS until,
       e.confidence AS confidence
ORDER BY e.start_time;

// ---------------------------------------------------------------------------
// 7. SPATIAL — everything that crossed a named line or region.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id})-[r:CROSSES]->(l:Location {region_id: $region_id})
RETURN o.global_id AS object, o.class AS class, o.color AS color, r.at AS at
ORDER BY r.at;

// ---------------------------------------------------------------------------
// 8. COUNTING — "how many of each kind of vehicle?" Counts distinct global
//    objects, so an object seen in five clips is still one vehicle: the whole
//    point of M7's cross-clip linking.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id})
RETURN o.class AS class, coalesce(o.vehicle_type, 'unknown') AS type,
       coalesce(o.color, 'unknown') AS color, count(*) AS n
ORDER BY n DESC, class, type;

// ---------------------------------------------------------------------------
// 9. CROSS-CLIP LINKING — objects M7 tracked across clip boundaries, which is
//    the evidence that global identity is doing real work.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id})-[s:SEEN_IN]->(c:Clip)
WITH o, count(DISTINCT c) AS n_clips, collect(c.clip_id) AS clips,
     collect(s.track_id) AS local_track_ids
WHERE n_clips > 1
RETURN o.global_id AS object, o.color AS color, n_clips, clips, local_track_ids
ORDER BY n_clips DESC;

// ---------------------------------------------------------------------------
// 10. SEMANTIC CORRECTION AUDIT — where the best-shot pass (M8) changed or
//     filled the clip-level answer. This is a paper result, queried directly
//     from the graph.
// ---------------------------------------------------------------------------
MATCH (o:TrafficObject {video_id: $video_id})-[r:HAS_ATTRIBUTE {winner: true}]->(a:AttributeValue)
WHERE r.source = 'best_shot'
RETURN o.global_id AS object, a.type AS attribute, a.value AS final_value,
       r.share AS clip_level_share, r.confidence AS confidence
ORDER BY o.global_id, a.type;
