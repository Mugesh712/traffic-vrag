// M10 — Traffic video knowledge graph schema.
//
// KEY DECISION: attributes are stored BOTH ways, on purpose.
//   * The winning canonical value is denormalized onto the object node as an
//     indexed property (o.color, o.vehicle_type, ...), because "all white
//     sedans" is a set-wide filter and deserves an index seek.
//   * The full evidence -- every candidate value with its vote share, plus
//     M8's confidence/source/uncertainty -- is normalized as
//     (:TrafficObject)-[:HAS_ATTRIBUTE]->(:AttributeValue), because inspecting
//     a distribution is a traversal from one known node and is cheap however
//     it is stored, and because a serialized JSON blob would be opaque to Cypher.
// The usual objection to denormalization (update anomalies) does not apply:
// this graph is a materialized view of immutable pipeline JSON, rebuilt by a
// single writer and never edited in place.
//
// IDENTIFIER SCOPE: global_id ("obj_0001") and event_id ("evt_00001") are only
// unique WITHIN a video, so uniqueness is enforced on a video-scoped `uid`
// composite, with the human-readable id kept alongside for display.
// AttributeValue is deliberately NOT video-scoped: a shared vocabulary node
// lets "which colours appear across all videos" be a single lookup.

// --- Uniqueness constraints (each also creates a backing index) -------------

CREATE CONSTRAINT video_id_unique IF NOT EXISTS
  FOR (v:Video) REQUIRE v.video_id IS UNIQUE;

CREATE CONSTRAINT clip_uid_unique IF NOT EXISTS
  FOR (c:Clip) REQUIRE c.uid IS UNIQUE;

CREATE CONSTRAINT frame_uid_unique IF NOT EXISTS
  FOR (f:Frame) REQUIRE f.uid IS UNIQUE;

CREATE CONSTRAINT object_uid_unique IF NOT EXISTS
  FOR (o:TrafficObject) REQUIRE o.uid IS UNIQUE;

CREATE CONSTRAINT event_uid_unique IF NOT EXISTS
  FOR (e:Event) REQUIRE e.uid IS UNIQUE;

CREATE CONSTRAINT location_uid_unique IF NOT EXISTS
  FOR (l:Location) REQUIRE l.uid IS UNIQUE;

CREATE CONSTRAINT attribute_value_key_unique IF NOT EXISTS
  FOR (a:AttributeValue) REQUIRE a.key IS UNIQUE;

// --- Indexes for the demo's dominant query patterns -------------------------

// "all white sedans" -- composite index, single seek.
CREATE INDEX object_color_type IF NOT EXISTS
  FOR (o:TrafficObject) ON (o.color, o.vehicle_type);

CREATE INDEX object_color IF NOT EXISTS
  FOR (o:TrafficObject) ON (o.color);

CREATE INDEX object_vehicle_type IF NOT EXISTS
  FOR (o:TrafficObject) ON (o.vehicle_type);

CREATE INDEX object_make IF NOT EXISTS
  FOR (o:TrafficObject) ON (o.make);

// Video-scoped deletes and per-video queries.
CREATE INDEX object_video IF NOT EXISTS FOR (o:TrafficObject) ON (o.video_id);
CREATE INDEX clip_video IF NOT EXISTS FOR (c:Clip) ON (c.video_id);
CREATE INDEX frame_video IF NOT EXISTS FOR (f:Frame) ON (f.video_id);
CREATE INDEX event_video IF NOT EXISTS FOR (e:Event) ON (e.video_id);
CREATE INDEX location_video IF NOT EXISTS FOR (l:Location) ON (l.video_id);

// "which vehicles overtook", "what happened before 12:05".
CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.type);
CREATE INDEX event_start IF NOT EXISTS FOR (e:Event) ON (e.start_time);

// Temporal filtering over frames.
CREATE INDEX frame_wallclock IF NOT EXISTS FOR (f:Frame) ON (f.wallclock_time);
CREATE INDEX frame_timestamp IF NOT EXISTS FOR (f:Frame) ON (f.video_timestamp_sec);

// Attribute-node lookups ("everything ever seen in silver").
CREATE INDEX attribute_type_value IF NOT EXISTS
  FOR (a:AttributeValue) ON (a.type, a.value);
