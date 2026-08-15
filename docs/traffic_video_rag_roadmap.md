# Semantically-Corrected Video Reasoning for Traffic Surveillance
## Complete Module-by-Module Build Roadmap

**Owner:** Mugesh | **Target:** Final-year research project + paper
**Suggested duration:** 16 weeks

---

## How to Read This

Each module has:
- **What it does** — the goal
- **Deliverable** — the concrete file/output that proves it's done
- **Model** — which Claude model to run in Claude Code
- **Claude Code prompt** — a starting prompt to paste into the terminal

**Model policy (set once, follow throughout):**

| Model | Use it for |
|---|---|
| **Sonnet 5** | Default. Feature code, wiring, debugging, refactors |
| **Opus 4.8** | Novel algorithms, schema design, retrieval fusion, paper-critical logic |
| **Haiku 4.5** | Boilerplate, docstrings, unit tests, config files, README updates |

> If you're on Claude Pro/Max, Opus costs nothing extra — use it freely for the four "contribution" modules (M4, M6, M7, M8). Those are what make the paper publishable.

---

## PHASE 0 — Foundation (Week 1)

### M0. Repo, Environment & Config Backbone

**What it does:** Sets up the project skeleton so every later module plugs in cleanly.

**Deliverable:**
```
traffic-vrag/
├── configs/          # YAML configs (thresholds, model paths)
├── src/
│   ├── ingest/
│   ├── perception/
│   ├── semantics/
│   ├── graph/
│   ├── retrieval/
│   ├── api/
│   └── utils/
├── data/{raw,clips,frames,crops,outputs}/
├── notebooks/
├── tests/
└── requirements.txt
```

**Model:** Haiku 4.5 (pure boilerplate)

**Claude Code prompt:**
```
Create a Python project skeleton for a video-to-knowledge-graph pipeline.
Use a config-driven design: all thresholds in configs/pipeline.yaml, loaded
via pydantic-settings. Add a logging util, a JSON schema module for per-clip
outputs, and a CLI entrypoint (typer) with stub subcommands: ingest, detect,
track, attribute, link, build-kg, serve.
```

**Critical rule from day one:** every module writes JSON to `data/outputs/` and reads JSON from the previous module. Never pass Python objects between stages. This is what makes the pipeline debuggable and reproducible — and the paper claims "per-clip JSONs and a master object index."

---

## PHASE 1 — Perception (Weeks 2–4)

### M1. Video Ingestion & Preprocessing

**What it does:** MP4 → clips → sampled frames with timestamps.

**Deliverable:** `data/clips/clip_000.mp4`, `data/frames/clip_000/frame_00123.jpg`, plus `clip_manifest.json` mapping every frame to a real wall-clock timestamp.

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/ingest/video_ingest.py using OpenCV. It should:
1. Split a long video into fixed-length clips (default 30s, configurable)
2. Sample frames at a configurable interval (0.5-1s)
3. Write a clip_manifest.json with: clip_id, frame_id, frame_path,
   video_timestamp_sec, wallclock_time, fps, resolution
Handle variable-fps videos correctly — compute timestamps from frame index
and actual fps, not assumed 30fps.
```

**Watch out:** timestamp accuracy matters more than you think. Every query in your demo ("before 12:05 PM") depends on this being right.

---

### M2. Object Detection — YOLOv11

**What it does:** Detects vehicles/persons per frame.

**Deliverable:** `detections/clip_000.json` — `[{frame_id, class, bbox, confidence}]`

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/perception/detector.py wrapping Ultralytics YOLOv11.
- Batch inference over a frame directory
- Filter to traffic classes: car, truck, bus, motorcycle, bicycle, person
- Configurable conf threshold and NMS IoU
- Write detections JSON per clip
- Support both CPU and CUDA, auto-detect
Include a --visualize flag that writes annotated frames for sanity checking.
```

**Dataset note:** Start with a public traffic dataset so your results are comparable — UA-DETRAC, BDD100K, or Cityflow. Don't start with random YouTube footage; reviewers will ask about the benchmark.

---

### M3. Multi-Object Tracking — ByteTrack + ReID

**What it does:** Turns per-frame boxes into persistent tracks with appearance embeddings.

**Deliverable:** `tracks/clip_000.json` — `[{track_id, class, frames[], bboxes[], centers[], velocity[], embedding}]`

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/perception/tracker.py integrating ByteTrack with an OSNet ReID
appearance extractor.
- Feed detections from M2, output per-track trajectories
- For each track, crop the object per frame and extract an OSNet embedding
- Store a rolling average embedding + keep the top-K highest-confidence crops
  per track (these become "best shots" later — critical for M8)
- Compute per-frame velocity and dominant direction of travel
- Write tracks JSON with all of the above
```

**This is where the "best shot" concept starts.** Save the sharpest, largest, least-occluded crops per track now — Module 8 depends entirely on them.

---

## PHASE 2 — The Novel Contributions (Weeks 5–8)

> These four modules are your paper. Use **Opus 4.8** and plan on iterating.

### M4. Deterministic, Gated Intra-Clip Association ⭐ *Contribution #3*

**What it does:** Fixes ID switches and track fragmentation *within* a clip using appearance gating and special IoU handling for stationary objects (parked cars, etc.).

**Deliverable:** `tracks_associated/clip_000.json` — merged, defragmented tracks.

**Model:** Opus 4.8

**Claude Code prompt:**
```
Design and implement src/perception/association.py — a deterministic
appearance-gated association algorithm that repairs ByteTrack output.

Requirements:
- Detect fragmented tracks: same object, multiple track_ids (occlusion gaps)
- Merge candidates must pass HARD GATES (all must hold):
  * class consistency
  * appearance cosine similarity > tau_app
  * temporal gap < max_gap_frames
  * motion plausibility (predicted position vs actual, using last velocity)
- Stationary-class handling: if a track's center variance is below a
  threshold, treat it as stationary and use IoU-centric matching instead
  of motion prediction
- Fully deterministic: same input always yields same output. No randomness.
- Emit a merge log (which IDs merged into which, and which gate scores) —
  I need this for the paper's ablation study.

Before writing code, propose the algorithm as pseudocode and explain the
gate ordering. I want to review it first.
```

**Ask Claude Code to explain before it codes.** You need to understand this deeply — it's a contribution you'll defend in a viva.

---

### M5. VLM Attribute Extraction

**What it does:** Runs a VLM on object crops to get color, type, make, model, direction.

**Deliverable:** `attributes_raw/clip_000.json` — per-track, per-frame attribute guesses with confidences.

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/semantics/vlm_extractor.py.
- Support pluggable backends: Florence-2, InternVL2, BLIP-2 (start with
  Florence-2, it's smallest)
- Input: object crops from tracks. Output: structured JSON attributes
  {color, vehicle_type, make, model, direction, confidence}
- Force structured output: use a constrained prompt that demands JSON only,
  and validate against a pydantic schema. Retry once on parse failure.
- Batch crops for throughput. Cache by crop hash so reruns are cheap.
- Sample N frames per track (not all) — configurable, default 8 spread
  across the track lifetime.
```

**Reality check:** VLM inference is the slowest part of your pipeline. Cache aggressively. On a laptop GPU, expect this to dominate your runtime.

---

### M6. Clip-Level Temporal Aggregation ⭐ *Contribution #2, Part 1*

**What it does:** Majority voting + confidence weighting across frames to produce one canonical attribute set per track.

**Deliverable:** `attributes_canonical/clip_000.json`

**Model:** Opus 4.8

**Claude Code prompt:**
```
Build src/semantics/temporal_voting.py.

For each track, aggregate per-frame VLM attributes into ONE canonical value
per attribute using confidence-weighted majority voting.

Requirements:
- Weight votes by: VLM confidence x crop quality (size, sharpness,
  occlusion estimate)
- Handle semantic near-duplicates: "white"/"off-white"/"silver-white"
  should map to a canonical vocabulary. Build a normalization map per
  attribute type.
- Emit vote distributions, not just winners (e.g. White 70%, Silver 20%,
  Gray 10%) — the paper's figure shows this
- Quality filtering: if the winning vote's margin is below a threshold OR
  total evidence is too weak, mark the attribute as "uncertain" rather
  than guessing
- Output canonical attributes + full vote distribution + confidence
```

**The "uncertain" flag matters.** Systems that admit uncertainty beat systems that confidently hallucinate — and that's literally your paper's thesis.

---

### M7. Semantically-Gated Cross-Clip Global Linking ⭐ *Contribution #4*

**What it does:** Links track IDs across clip boundaries so `Vehicle_12` in clip 3 is the same car as `Vehicle_47` in clip 4.

**Deliverable:** `master_object_index.json` — global object IDs mapping to all their per-clip sightings.

**Model:** Opus 4.8

**Claude Code prompt:**
```
Build src/semantics/global_linking.py — cross-clip re-identification.

Link tracks across consecutive clips into global object IDs. A candidate
link must pass ALL hard gates:
  1. CLASS gate: same object class
  2. SEMANTIC gate: canonical attributes compatible (a white sedan cannot
     become a blue truck). Allow "uncertain" to match anything.
  3. MOTION gate: exit position/velocity at end of clip A must plausibly
     reach entry position at start of clip B, given the time gap
  4. APPEARANCE gate: ReID embedding cosine similarity > threshold

Then score surviving candidates and solve as a bipartite assignment
(Hungarian algorithm) — not greedy matching.

Output a master_object_index.json:
  global_id -> [{clip_id, track_id, first_seen, last_seen, gate_scores}]

Emit rejected-link logs with which gate failed — needed for ablations.
```

**This is your hardest module.** Budget a full week. Test it on a video where you manually know the ground truth for 5–10 vehicles.

---

### M8. Global Best-Shot VLM Confirmation ⭐ *Contribution #2, Part 2*

**What it does:** Final semantic pass — takes the single best crop of each *global* object and re-runs the VLM to lock in canonical attributes.

**Deliverable:** `global_objects_final.json`

**Model:** Opus 4.8 (for design), Sonnet 5 (for implementation)

**Claude Code prompt:**
```
Build src/semantics/best_shot_confirmation.py.

For each global object from M7:
1. Select the best-shot crop across ALL its clips. Score crops by:
   resolution, sharpness (Laplacian variance), detection confidence,
   occlusion estimate, and how frontal/side-on the view is.
2. Run the VLM on the top-3 best shots with a high-detail prompt
3. Reconcile the result against the aggregated clip-level canonical
   attributes from M6:
   - If they agree -> confirm, boost confidence
   - If they disagree -> best-shot wins for fine-grained attributes
     (make/model), clip-level voting wins for coarse ones (color/type)
   - Log every correction made
4. Write global_objects_final.json with confirmed canonical attributes

The correction log is a paper result — it quantifies how much semantic
correction the system actually performs.
```

---

## PHASE 3 — Knowledge Graph (Weeks 9–10)

### M9. Event Generation

**What it does:** Derives events (overtake, stop, turn, lane change, park) from trajectories.

**Deliverable:** `events.json`

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/graph/event_detector.py — rule-based event detection from
global object trajectories.

Detect:
- OVERTAKE: object A behind B, then ahead of B, same direction, within
  a time window, lateral displacement present
- STOP: velocity near zero for > N seconds
- PARK: stopped and remains stopped through end of observation
- TURN: heading change > 45 degrees sustained
- LANE_CHANGE: lateral displacement > threshold without heading change
- CROSSES: trajectory intersects a defined region/line

Each event: {event_id, type, subject_id, object_id (nullable),
start_time, end_time, confidence, evidence_frames[]}

Make regions/lines configurable per-video in a YAML so the same code
works on different camera angles.
```

**Do rules first, LLM later.** Rules are deterministic, explainable, and fast. Add an optional LLM event-describer only after rules work.

---

### M10. Knowledge Graph Construction — Neo4j

**What it does:** Materializes everything into a queryable graph.

**Deliverable:** A populated Neo4j database + `src/graph/schema.cypher`

**Model:** Opus 4.8 (schema design), Sonnet 5 (loader code)

**Claude Code prompt:**
```
First: design the Neo4j schema for a traffic video knowledge graph.

Node types: Vehicle, Person, Frame, Location, Event, Clip, Video
Relationships: HAS_ATTRIBUTE, SEEN_AT, MOVED_TO, INTERACTS_WITH,
OVERTAKES, STOPS_AT, PARKED_AT, CROSSES, PART_OF

Key design question I need you to reason about: should attributes be
node properties or separate Attribute nodes? Argue both sides given that
I need to query "all white sedans" efficiently AND store confidence +
vote distributions per attribute.

Then write:
- schema.cypher with constraints and indexes
- src/graph/kg_builder.py that ingests global_objects_final.json +
  events.json into Neo4j idempotently (re-running must not duplicate)
- A set of 10 example Cypher queries covering the query types in my demo
```

---

## PHASE 4 — Retrieval & Reasoning (Weeks 11–12)

### M11. Vector Store & Object Timeline Embeddings

**What it does:** Makes objects semantically searchable.

**Deliverable:** Populated ChromaDB collection.

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/retrieval/vector_store.py using ChromaDB.

For each global object, generate a natural-language "timeline summary"
string, e.g.:
"White Toyota sedan, first seen 11:02:10 heading east, overtook a blue
truck at 11:04:20, stopped at intersection 11:05:42."

Embed these summaries and store with metadata: global_id, class, color,
make, type, first_seen, last_seen, event_types[].

Also embed individual events separately as a second collection.
Support metadata-filtered vector search (e.g. color=white AND
event_type=OVERTAKE).
```

You've done this in HealthOps AI — this module should go fast.

---

### M12. Hybrid Retrieval (Vector + Graph) ⭐

**What it does:** Given a question, pulls the right subgraph.

**Deliverable:** `src/retrieval/hybrid_retriever.py`

**Model:** Opus 4.8

**Claude Code prompt:**
```
Build src/retrieval/hybrid_retriever.py.

Pipeline:
1. QUERY UNDERSTANDING: use an LLM to parse the question into a structured
   intent: {target_attributes, event_types, time_range, relations,
   question_type: factual|counterfactual|forecast|counting}
2. VECTOR RETRIEVAL: semantic search over object timelines with metadata
   filters derived from step 1
3. GRAPH RETRIEVAL: generate a parameterized Cypher query from the
   structured intent (use templates, NOT free-form LLM Cypher generation —
   free-form is unreliable and unsafe)
4. FUSION: merge results, dedupe by global_id, rank by combined score
5. CONTEXT ASSEMBLY: for top-K objects, pull full timelines, attributes,
   events, and evidence frame paths from Neo4j

Return a structured RetrievalResult, not a string blob.

Reason carefully about fusion ranking before coding — explain your
approach first.
```

**Templated Cypher over generated Cypher.** This is the single most important engineering decision in this module. LLM-generated Cypher fails unpredictably; templates are reliable and defensible in a paper.

---

### M13. LLM Reasoning & Explainable Answer

**What it does:** Turns retrieved context into a grounded, cited answer.

**Deliverable:** `src/retrieval/answer_generator.py`

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/retrieval/answer_generator.py.

Take a RetrievalResult and produce:
{
  answer: str,
  supporting_object_ids: [str],
  timestamps: [{start, end}],
  evidence_frames: [path],
  kg_subgraph: {nodes, edges},
  reasoning_trace: str
}

Requirements:
- The prompt must instruct the model to answer ONLY from provided context
  and say "insufficient evidence" when the KG doesn't support an answer
- Every claim must cite a global_id and timestamp
- Support local models via Ollama (Qwen2.5, Llama 3.1, Mistral, Phi-4)
  with a pluggable backend interface
- Include the kg_subgraph so the UI can render the explanation
```

---

## PHASE 5 — Product (Weeks 13–14)

### M14. FastAPI Backend

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build src/api/main.py with FastAPI:
POST /upload           -> accept video, return job_id
GET  /jobs/{id}/status -> pipeline progress per stage
POST /query            -> {job_id, question} -> full answer object
GET  /objects/{gid}    -> object detail + timeline + crops
GET  /frames/{path}    -> serve evidence images

Run the pipeline as a background job with per-stage progress reporting.
Use a simple SQLite job store. Add CORS for the React frontend.
```

### M15. React Frontend

**Model:** Sonnet 5

**Claude Code prompt:**
```
Build a React demo UI:
- Video upload with drag-drop + pipeline progress (per-stage bars)
- Chat-style question box
- Answer panel: text answer, object ID chips, timeline strip with
  timestamps, evidence image gallery, and a KG subgraph visualization
- Object explorer: browse all detected objects with their canonical
  attributes and confidence

Use Tailwind. Use react-force-graph for the KG subgraph.
Design it clean and technical — this goes in my paper's figures.
```

---

## PHASE 6 — Research Output (Weeks 15–16)

### M16. Evaluation & Benchmarking ⭐ *This is what makes it a paper*

**Model:** Opus 4.8

**Claude Code prompt:**
```
Build an evaluation harness (src/eval/) that measures:

TRACKING/ID METRICS:
- IDF1, MOTA, ID switches, track fragmentation
- Compare: raw ByteTrack vs +M4 association vs +M7 global linking

ATTRIBUTE METRICS:
- Attribute accuracy vs manual ground truth
- Attribute consistency rate (does the same object keep the same color?)
- Compare: raw per-frame VLM vs +M6 voting vs +M8 best-shot confirmation

QA METRICS:
- Build a QA benchmark set: factual, counting, temporal, counterfactual,
  forecasting questions with ground-truth answers
- Accuracy vs baselines: (a) a Video-LLM end-to-end, (b) caption-based RAG

ABLATIONS (one per contribution):
- Remove appearance gating from M4
- Remove semantic gate from M7
- Remove best-shot confirmation from M8
- Replace KG retrieval with plain vector retrieval

Output results as LaTeX-ready tables.
```

**Start annotating ground truth NOW, in parallel with Phase 2.** Manual annotation is the bottleneck that kills student research timelines. Even 20 vehicles across 5 minutes of video is enough to publish.

### M17. Docker + Reproducibility

**Model:** Haiku 4.5

```
Write docker-compose.yml with services: neo4j, chromadb, api, frontend.
Add a Dockerfile for the pipeline with CUDA base image. Write a
reproduce.sh that runs the full pipeline end-to-end on a sample video.
```

---

## Timeline Summary

| Weeks | Phase | Modules |
|---|---|---|
| 1 | Foundation | M0 |
| 2–4 | Perception | M1, M2, M3 |
| 5–8 | **Contributions** | M4, M5, M6, M7, M8 |
| 9–10 | Knowledge Graph | M9, M10 |
| 11–12 | Retrieval & RAG | M11, M12, M13 |
| 13–14 | Product | M14, M15 |
| 15–16 | Research Output | M16, M17 |

---

## Six Rules That Will Save Your Project

1. **JSON between every stage.** Never pass objects. You'll thank yourself when Module 7 breaks and you need to inspect Module 6's output.

2. **Version your outputs.** `outputs/v1/`, `outputs/v2/`. When you change a threshold, you need to compare against the old run.

3. **Build a 30-second test video and use it constantly.** Full-pipeline runs on long video will take hours. Iterate on 30 seconds.

4. **Annotate ground truth in parallel with Phase 2, not at the end.** This is the #1 timeline killer.

5. **Ask Claude Code to explain algorithms before coding them** for M4, M6, M7, M8. You have to defend these in a viva.

6. **Write the paper sections as you finish each module** while the details are fresh. Methods sections written 3 months later are vague and wrong.

---

## Where to Use Which Model — Quick Card

```
Opus 4.8   -> M4, M6, M7, M8 (design), M10 (schema), M12, M16
Sonnet 5   -> M1, M2, M3, M5, M8 (impl), M9, M10 (loader),
              M11, M13, M14, M15
Haiku 4.5  -> M0, M17, tests, docstrings, READMEs
```

---

## Immediate Next Steps

1. Pick your dataset (UA-DETRAC or BDD100K) and download 2–3 clips
2. Run M0 — get the skeleton up today
3. Get M1→M2→M3 working end-to-end on one 30-second clip this week
4. Start ground-truth annotation for 10 vehicles in parallel

Once M3 produces clean tracks on a short clip, everything downstream becomes tractable.
