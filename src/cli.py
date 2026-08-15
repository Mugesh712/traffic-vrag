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
    settings = get_settings()
    logger.info("attribute: clip_id=%s backend=%s", clip_id, settings.vlm.backend)
    raise NotImplementedError("M5: src/semantics/vlm_extractor.py not implemented yet")


@app.command()
def link(
    video_id: str = typer.Argument(..., help="Video ID to link tracks across clips for."),
) -> None:
    """Run cross-clip global linking to build the master object index (M7)."""
    settings = get_settings()
    logger.info("link: video_id=%s appearance_threshold=%s", video_id, settings.linking.appearance_similarity_threshold)
    raise NotImplementedError("M7: src/semantics/global_linking.py not implemented yet")


@app.command(name="build-kg")
def build_kg(
    video_id: str = typer.Argument(..., help="Video ID to materialize into the knowledge graph."),
) -> None:
    """Load the master object index and events into Neo4j (M10)."""
    logger.info("build-kg: video_id=%s", video_id)
    raise NotImplementedError("M10: src/graph/kg_builder.py not implemented yet")


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
