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
    visualize: bool = typer.Option(
        False, "--visualize", help="Write annotated frames for sanity checking."
    ),
    batch_size: int = typer.Option(16, help="Frames per inference batch."),
) -> None:
    """Run object detection on a clip's sampled frames (M2)."""
    from src.perception.detector import detect_clip

    settings = get_settings()
    clip_detections = detect_clip(
        clip_id, settings=settings, batch_size=batch_size, visualize=visualize
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
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the API server to."),
    port: int = typer.Option(8000, help="Port to bind the API server to."),
) -> None:
    """Start the FastAPI backend (M14)."""
    logger.info("serve: host=%s port=%s", host, port)
    raise NotImplementedError("M14: src/api/main.py not implemented yet")


if __name__ == "__main__":
    app()
