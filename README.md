# Traffic-VRAG

Semantically-corrected video reasoning for traffic surveillance: a video goes
in, a queryable knowledge graph of globally-identified vehicles comes out, and
questions are answered with citations back to the frames that support them.

The pipeline's organising idea is that **admitting uncertainty beats confident
hallucination**. Attributes the system is not sure about are marked uncertain
rather than guessed, and that flag is honoured everywhere downstream — an
uncertain value cannot veto a correct cross-clip link, and answers decline
rather than invent.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| M1 | `src/ingest/video_ingest.py` | Video → clips → sampled frames with real timestamps |
| M2 | `src/perception/detector.py` | YOLOv11 detection, traffic classes only |
| M3 | `src/perception/tracker.py` | ByteTrack + OSNet ReID → per-clip tracks |
| M4 | `src/perception/association.py` | Gated intra-clip association (repairs fragmentation) |
| M5 | `src/semantics/vlm_extractor.py` | Florence-2 attributes per crop |
| M6 | `src/semantics/temporal_voting.py` | Confidence-weighted voting → canonical attributes |
| M7 | `src/semantics/global_linking.py` | Semantically-gated cross-clip identity (Hungarian) |
| M8 | `src/semantics/best_shot_confirmation.py` | High-detail re-read of each object's best crops |
| M9 | `src/graph/event_detector.py` | Rule-based events (stop, park, turn, overtake, …) |
| M10 | `src/graph/kg_builder.py` | Neo4j knowledge graph |
| M11 | `src/retrieval/vector_store.py` | ChromaDB timeline + event embeddings |
| M12 | `src/retrieval/hybrid_retriever.py` | Vector + templated-Cypher retrieval, rank fusion |
| M13 | `src/retrieval/answer_generator.py` | Grounded answers with validated citations |
| M14 | `src/api/main.py` | FastAPI backend |
| M15 | `frontend/` | React demo UI |
| M16 | `src/eval/` | Metrics, ablations, LaTeX tables |
| M17 | `Dockerfile`, `docker-compose.yml`, `scripts/reproduce.sh` | Reproducibility |

## Quick start

### Reproduce end to end

```bash
./scripts/reproduce.sh          # downloads a real clip, runs M1→M11 + evaluation
```

Stages needing a service that is not running (Neo4j for the graph) are skipped
with a warning rather than failing the run.

### Full stack in Docker

```bash
docker compose up -d
docker compose exec api ollama --version   # LLM lives in the ollama service
./scripts/reproduce.sh --docker
```

Then open the UI at <http://localhost:5173>, Neo4j Browser at
<http://localhost:7474>.

First run downloads ~1GB of model weights into a named volume; subsequent runs
reuse it.

### Locally, without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli --help
```

## Configuration

Everything lives in `configs/pipeline.yaml`. Precedence is **environment
variables → YAML → defaults**, so nothing needs editing to point at different
services:

```bash
export PIPELINE__NEO4J__PASSWORD=…        # keep secrets out of the repo
export PIPELINE__DETECT__CONF_THRESHOLD=0.4
```

## Evaluation

```bash
.venv/bin/python -m src.cli evaluate car_detection              # tables that need no annotation
.venv/bin/python -m src.cli evaluate car_detection --run-qa     # + QA vs baselines (needs an LLM)
```

Metrics that require ground truth (MOTA, IDF1, attribute accuracy) are
reported as **not available** until annotation exists, never as zeros. To
annotate:

```bash
.venv/bin/python -m src.cli evaluate car_detection --write-template     # boxes/attributes
.venv/bin/python -m src.cli evaluate car_detection --write-qa-template  # questions
```

Both emit a template that must be corrected by hand; the loader refuses a file
still carrying its unreviewed marker, because scoring the system against its
own proposals would report ~100% and mean nothing.

**Faster route — import a benchmark's own labels.** Public tracking datasets
already ship human annotations, so the tracking tables need no manual boxing
at all:

```bash
.venv/bin/python -m src.cli import-gt <video_id> path/to/gt.txt --format mot
.venv/bin/python -m src.cli import-gt <video_id> path/to/MVI_20011.xml --format detrac
```

The importer intersects annotations with the frames M1 actually sampled and
reports what it dropped. This matters: benchmarks label *every* video frame,
so importing them wholesale would score each unsampled frame as a miss and
make MOTA measure the sampling rate rather than the tracker. UA-DETRAC XML
also carries `vehicle_type`, which seeds part of the attribute ground truth —
leaving colour as the main thing still needing a human.

## Known limitations

These are real and measured, not hypothetical:

- **Overhead viewpoint hurts detection.** On near-nadir traffic-camera
  footage, COCO-trained YOLO under-detects badly (`yolo11n` 0/11 frames,
  `yolo11s` 1/11, `yolo11m` 5/11 on a hand-checked sample). Fine-tuning on an
  aerial vehicle dataset, or preferring oblique camera angles, is the fix.
- **Sampling interval is now 0.5s, not 1.0s** (resolved, was an open
  decision). At 1.0s ByteTrack confirmed *zero* tracks and the pipeline
  silently produced nothing. Measured tracks by interval on a real clip:
  `1.0 → 0`, `0.5 → 4`, `0.25 → 7`, `0.16 → 8`. VLM cost turned out to scale
  with track count, not frame count (M5 caps crops per track), so denser
  sampling is far cheaper than it first appeared; `reproduce.sh` uses 0.16s to
  recover more objects for evaluation.
- **Cross-clip linking is unexercised** on the sample clip: every object
  appears in one clip, so M7 has nothing to link. Longer footage is needed to
  evaluate it.
- **Small local LLMs under-decline.** `qwen2.5:1.5b` sometimes answers an
  unanswerable question instead of abstaining; the roadmap's 7B suggestion
  should be re-measured before publication.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 353 tests
```
