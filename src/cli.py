"""CLI entrypoint for the traffic video RAG pipeline.

Run with: python -m src.cli <command> --help
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import typer

from src.utils.config import get_settings
from src.utils.logging import get_logger

app = typer.Typer(help="Traffic video RAG pipeline CLI.")
logger = get_logger(__name__)


@app.command()
def ingest(
    video_path: str = typer.Argument(..., help="Path to the source video file."),
    clip_length_sec: Optional[float] = typer.Option(
        None, help="Override configs/pipeline.yaml ingest.clip_length_sec."
    ),
    frame_interval_sec: Optional[float] = typer.Option(
        None, help="Override configs/pipeline.yaml ingest.frame_sample_interval_sec."
    ),
    start_time: Optional[str] = typer.Option(
        None,
        help="ISO-8601 wall-clock time of the video's first frame "
        "(e.g. 2026-08-15T10:00:00). Defaults to now.",
    ),
) -> None:
    """Split a video into clips and sampled frames (M1)."""
    from src.ingest.video_ingest import ingest_video

    settings = get_settings()
    start_wallclock = datetime.fromisoformat(start_time) if start_time else None
    manifest = ingest_video(
        video_path,
        settings=settings,
        clip_length_sec=clip_length_sec,
        frame_sample_interval_sec=frame_interval_sec,
        start_wallclock=start_wallclock,
    )
    typer.echo(
        f"Ingested {manifest.video_id}: {len(manifest.clips)} clips, "
        f"{sum(len(c.frames) for c in manifest.clips)} sampled frames"
    )


@app.command()
def detect(
    clip_id: str = typer.Argument(..., help="Clip ID to run detection on."),
    video_id: str = typer.Argument(
        ..., help="Video ID the clip belongs to (frames are stored per video)."
    ),
    visualize: bool = typer.Option(
        False, "--visualize", help="Write annotated frames for sanity checking."
    ),
    batch_size: int = typer.Option(16, help="Frames per inference batch."),
) -> None:
    """Run object detection on a clip's sampled frames (M2)."""
    from src.perception.detector import detect_clip

    settings = get_settings()
    clip_detections = detect_clip(
        clip_id, video_id, settings=settings, batch_size=batch_size, visualize=visualize
    )
    typer.echo(f"Detected {len(clip_detections.detections)} objects in {clip_id}")


@app.command()
def track(
    clip_id: str = typer.Argument(..., help="Clip ID to run tracking on."),
) -> None:
    """Run multi-object tracking with ReID on a clip's detections (M3)."""
    from src.perception.tracker import track_clip

    settings = get_settings()
    clip_tracks = track_clip(clip_id, settings=settings)
    typer.echo(f"Tracked {len(clip_tracks.tracks)} objects in {clip_id}")


@app.command()
def associate(
    clip_id: str = typer.Argument(..., help="Clip ID to repair fragmented tracks for."),
) -> None:
    """Repair intra-clip track fragmentation with gated association (M4)."""
    from src.perception.association import associate_clip

    settings = get_settings()
    associated = associate_clip(clip_id, settings=settings)
    typer.echo(
        f"Associated {clip_id}: {len(associated.tracks)} tracks "
        f"after {len(associated.merge_log)} merges"
    )


@app.command()
def attribute(
    clip_id: str = typer.Argument(..., help="Clip ID to extract attributes for."),
) -> None:
    """Run VLM attribute extraction on a clip's tracks (M5)."""
    from src.semantics.vlm_extractor import extract_clip_attributes

    settings = get_settings()
    clip_attributes = extract_clip_attributes(clip_id, settings=settings)
    typer.echo(f"Extracted attributes for {len(clip_attributes.attributes)} frame-observations in {clip_id}")


@app.command()
def vote(
    clip_id: str = typer.Argument(..., help="Clip ID to aggregate attributes for."),
) -> None:
    """Aggregate per-frame attributes into canonical values by voting (M6)."""
    from src.semantics.temporal_voting import vote_clip_attributes

    settings = get_settings()
    canonical = vote_clip_attributes(clip_id, settings=settings)
    n_uncertain = sum(v.uncertain for t in canonical.tracks for v in t.votes)
    typer.echo(
        f"Voted canonical attributes for {len(canonical.tracks)} tracks in {clip_id} "
        f"({n_uncertain} attributes marked uncertain)"
    )


@app.command()
def link(
    video_id: str = typer.Argument(..., help="Video ID to link tracks across clips for."),
) -> None:
    """Run cross-clip global linking to build the master object index (M7)."""
    from src.semantics.global_linking import link_video

    settings = get_settings()
    index = link_video(video_id, settings=settings)
    n_multi_clip = sum(1 for o in index.objects if len(o.sightings) > 1)
    typer.echo(
        f"Linked {video_id}: {len(index.objects)} global objects "
        f"({n_multi_clip} spanning multiple clips)"
    )


@app.command()
def confirm(
    video_id: str = typer.Argument(..., help="Video ID to run best-shot confirmation for."),
) -> None:
    """Re-read each global object's best shots and lock in attributes (M8)."""
    from src.semantics.best_shot_confirmation import confirm_video

    settings = get_settings()
    final_index = confirm_video(video_id, settings=settings)
    corrected = sum(1 for e in final_index.correction_log if e.outcome == "corrected")
    filled = sum(1 for e in final_index.correction_log if e.outcome == "filled")
    typer.echo(
        f"Confirmed {len(final_index.objects)} objects in {video_id} "
        f"({filled} attributes filled, {corrected} corrected)"
    )


@app.command()
def events(
    video_id: str = typer.Argument(..., help="Video ID to detect events for."),
) -> None:
    """Detect rule-based events from global object trajectories (M9)."""
    from src.graph.event_detector import detect_events

    settings = get_settings()
    log = detect_events(video_id, settings=settings)
    typer.echo(f"Detected {len(log.events)} events for {video_id}")


@app.command(name="build-kg")
def build_kg(
    video_id: str = typer.Argument(..., help="Video ID to materialize into the knowledge graph."),
) -> None:
    """Load the master object index and events into Neo4j (M10)."""
    from src.graph.kg_builder import build_kg as _build_kg

    settings = get_settings()
    payload = _build_kg(video_id, settings=settings)
    counts = payload.counts()
    typer.echo(
        f"Loaded {video_id} into Neo4j: "
        + ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    )


@app.command()
def index(
    video_id: str = typer.Argument(..., help="Video ID to embed into the vector store."),
) -> None:
    """Embed object timelines and events into ChromaDB (M11)."""
    from src.retrieval.vector_store import populate_vector_store

    settings = get_settings()
    object_ids, event_ids = populate_vector_store(video_id, settings=settings)
    typer.echo(f"Indexed {video_id}: {len(object_ids)} object timelines, {len(event_ids)} events")


@app.command()
def query(
    video_id: str = typer.Argument(..., help="Video ID to query."),
    question: str = typer.Argument(..., help="Natural-language question."),
    show_cypher: bool = typer.Option(False, "--show-cypher", help="Print the templated Cypher used."),
) -> None:
    """Answer a question with hybrid vector + graph retrieval (M12)."""
    from src.retrieval.hybrid_retriever import hybrid_retrieve

    settings = get_settings()
    result = hybrid_retrieve(question, video_id, settings=settings)

    for warning in result.warnings:
        typer.echo(f"warning: {warning}")

    typer.echo(f"intent: {result.intent.question_type}", err=False)
    if result.intent.target_attributes:
        typer.echo(f"  attributes: {result.intent.target_attributes}")
    if result.intent.event_types:
        typer.echo(f"  events: {result.intent.event_types}")
    if not result.intent.time_range.is_empty():
        typer.echo(f"  time: {result.intent.time_range.after} .. {result.intent.time_range.before}")

    if result.count is not None:
        typer.echo(f"\ncount: {result.count}")
        for row in result.count_breakdown:
            typer.echo(f"  {row}")

    typer.echo(f"\n{len(result.objects)} object(s):")
    for obj in result.objects:
        typer.echo(
            f"  {obj.global_id}  score={obj.score:.3f}  via={'+'.join(obj.sources)}  {obj.timeline_summary}"
        )

    if show_cypher:
        for cypher in result.cypher_queries:
            typer.echo(f"\n--- cypher ---\n{cypher}")


@app.command()
def ask(
    video_id: str = typer.Argument(..., help="Video ID to query."),
    question: str = typer.Argument(..., help="Natural-language question."),
) -> None:
    """Retrieve and answer a question with a grounded, cited answer (M12+M13)."""
    from src.retrieval.answer_generator import generate_answer
    from src.retrieval.hybrid_retriever import hybrid_retrieve

    settings = get_settings()
    result = hybrid_retrieve(question, video_id, settings=settings)
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")

    answer = generate_answer(result, settings=settings)
    typer.echo(f"\n{answer.answer}")
    typer.echo(f"\nstatus: {answer.status}")
    if answer.supporting_object_ids:
        typer.echo(f"supporting objects: {answer.supporting_object_ids}")
    if answer.timestamps:
        typer.echo(f"timestamps: {[(t.start, t.end) for t in answer.timestamps]}")
    if answer.unsupported_citations:
        typer.echo(f"unsupported citations: {answer.unsupported_citations}")
    typer.echo(f"reasoning: {answer.reasoning_trace}")


@app.command()
def evaluate(
    video_id: str = typer.Argument(..., help="Video ID to evaluate."),
    write_template: bool = typer.Option(
        False, "--write-template",
        help="Emit a ground-truth annotation template pre-filled with detections, then exit.",
    ),
    write_qa_template: bool = typer.Option(
        False, "--write-qa-template", help="Emit a starter QA benchmark template, then exit."
    ),
    run_qa: bool = typer.Option(
        False, "--run-qa",
        help="Run the QA benchmark (needs a live LLM; add --no-baselines to skip the comparisons).",
    ),
    baselines: bool = typer.Option(True, help="Include the caption-RAG and frame-caption baselines."),
    overwrite: bool = typer.Option(False, help="Allow a --write-*-template to replace an existing file."),
) -> None:
    """Run the evaluation harness and emit LaTeX-ready tables (M16)."""
    from src.eval.ground_truth import write_annotation_template
    from src.eval.qa import write_benchmark_template
    from src.eval.report import render_text
    from src.eval.runner import evaluate_video

    settings = get_settings()

    if write_template:
        path = write_annotation_template(video_id, settings=settings, overwrite=overwrite)
        typer.echo(
            f"Wrote annotation template -> {path}\n"
            "Correct the boxes and attributes by hand, delete the "
            "REMOVE_THIS_KEY_ONCE_REVIEWED key, then re-run `evaluate`."
        )
        return

    if write_qa_template:
        path = write_benchmark_template(video_id, settings=settings, overwrite=overwrite)
        typer.echo(
            f"Wrote QA benchmark template -> {path}\n"
            "Replace the questions with ones about your footage, fill in the expected "
            "answers by hand, delete the REMOVE_THIS_KEY_ONCE_REVIEWED key, then re-run "
            "with --run-qa."
        )
        return

    tables, latex_path = evaluate_video(
        video_id, settings=settings, run_qa=run_qa, include_baselines=baselines
    )
    typer.echo(render_text(tables))
    typer.echo(f"LaTeX tables -> {latex_path}")


@app.command(name="import-gt")
def import_gt(
    video_id: str = typer.Argument(..., help="Video ID the annotations belong to."),
    source: str = typer.Argument(..., help="Path to the annotation file (MOT gt.txt or UA-DETRAC XML)."),
    fmt: str = typer.Option("mot", "--format", help="Source format: 'mot' or 'detrac'."),
    frame_offset: int = typer.Option(
        -1, help="Added to source frame numbers to reach this pipeline's 0-based index."
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing ground-truth file."),
) -> None:
    """Convert benchmark annotations into ground truth (M16)."""
    from src.eval.import_gt import import_ground_truth

    settings = get_settings()
    report = import_ground_truth(
        video_id, source, fmt, settings=settings, frame_offset=frame_offset, overwrite=overwrite
    )
    typer.echo(report.summary())
    if report.n_tracks == 0:
        typer.echo(
            "\nNothing survived frame alignment. The most likely cause is a frame-index "
            "mismatch: try --frame-offset 0 if the source counts from zero."
        )


@app.command()
def serve(
    host: str = typer.Option(None, help="Host to bind the API server to (default: configs/pipeline.yaml)."),
    port: int = typer.Option(None, help="Port to bind the API server to (default: configs/pipeline.yaml)."),
) -> None:
    """Start the FastAPI backend (M14)."""
    import uvicorn

    settings = get_settings()
    host = host or settings.api.host
    port = port or settings.api.port
    logger.info("serve: host=%s port=%s", host, port)
    uvicorn.run("src.api.main:app", host=host, port=port)


if __name__ == "__main__":
    app()
