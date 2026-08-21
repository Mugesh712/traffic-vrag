"""M14 — Runs the full M1-M11 pipeline as one background job, reporting
progress per stage to the SQLite job store.

video_id == job_id, BY CONSTRUCTION. The uploaded file is saved as
`<job_id><ext>` before M1 ever sees it, and M1 derives video_id from the
file's stem -- so the two identifiers are the same string, not just linked by
a foreign key. That removes an entire class of "which video does this job
belong to" bugs and lets every downstream lookup (`/objects`, `/query`) use
job_id directly as the video_id the rest of the pipeline already understands.

BUILD_KG DEGRADES, THE REST DOES NOT. M12's retriever already tolerates a
missing Neo4j (vector-only, with a warning) -- so if the graph is unreachable
when a job runs, that is a reduced-functionality outcome, not a fatal one.
Every other stage is left to fail the job outright: a broken detector or a
missing attributes file is a real defect, not an optional integration.
"""
from __future__ import annotations

from pathlib import Path

from src.api.job_store import JobStore
from src.graph.event_detector import detect_events
from src.graph.kg_builder import build_kg
from src.ingest.video_ingest import ingest_video
from src.perception.association import associate_clip
from src.perception.detector import detect_clip
from src.perception.tracker import track_clip
from src.retrieval.vector_store import populate_vector_store
from src.semantics.best_shot_confirmation import confirm_video
from src.semantics.global_linking import link_video
from src.semantics.temporal_voting import vote_clip_attributes
from src.semantics.vlm_extractor import extract_clip_attributes
from src.utils.config import PipelineSettings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# (stage name, per-clip function, needs_video_id). Run in this order for every
# clip before moving to the next stage, matching each stage's own file-based
# dependency on the previous one (M3 reads M2's output, M4 reads M3's, ...).
#
# Only M2 takes video_id: it locates frames on disk, which are namespaced per
# video. The later stages read the previous stage's JSON, which already carries
# the frame paths it resolved.
_PER_CLIP_STAGES = (
    ("detect", detect_clip, True),
    ("track", track_clip, False),
    ("associate", associate_clip, False),
    ("attribute", extract_clip_attributes, False),
    ("vote", vote_clip_attributes, False),
)


def run_pipeline(job_id: str, video_path: str, settings: PipelineSettings, store: JobStore) -> None:
    """Entry point for the background task. Never raises: failures are
    recorded on the job row so the API layer never sees an unhandled
    exception from a background thread."""
    try:
        store.mark_stage_started(job_id, "ingest")
        manifest = ingest_video(video_path, settings=settings)
        # video_id is already job_id (see module docstring), but read it back
        # from the manifest rather than assuming, so a naming mismatch would
        # surface as a loud downstream KeyError instead of a silent one.
        video_id = manifest.video_id
        store.set_video_id(job_id, video_id)
        store.mark_stage_completed(job_id, "ingest")

        clip_ids = [clip.clip_id for clip in manifest.clips]
        for stage_name, stage_fn, needs_video_id in _PER_CLIP_STAGES:
            for i, clip_id in enumerate(clip_ids, start=1):
                store.mark_stage_started(job_id, stage_name, detail=f"clip {i}/{len(clip_ids)}: {clip_id}")
                if needs_video_id:
                    stage_fn(clip_id, video_id, settings=settings)
                else:
                    stage_fn(clip_id, settings=settings)
            store.mark_stage_completed(job_id, stage_name)

        store.mark_stage_started(job_id, "link")
        link_video(video_id, settings=settings)
        store.mark_stage_completed(job_id, "link")

        store.mark_stage_started(job_id, "confirm")
        confirm_video(video_id, settings=settings)
        store.mark_stage_completed(job_id, "confirm")

        store.mark_stage_started(job_id, "events")
        detect_events(video_id, settings=settings)
        store.mark_stage_completed(job_id, "events")

        store.mark_stage_started(job_id, "build_kg")
        try:
            build_kg(video_id, settings=settings)
            store.mark_stage_completed(job_id, "build_kg")
        except Exception as exc:
            logger.warning(
                "run_pipeline: job=%s build_kg failed, continuing without the graph "
                "(hybrid retrieval degrades to vector-only): %s", job_id, exc,
            )
            store.mark_stage_completed(job_id, "build_kg", detail=f"skipped: {exc}")

        store.mark_stage_started(job_id, "index")
        populate_vector_store(video_id, settings=settings)
        store.mark_stage_completed(job_id, "index")

        store.mark_completed(job_id)
        logger.info("run_pipeline: job=%s video_id=%s completed", job_id, video_id)
    except Exception as exc:
        logger.exception("run_pipeline: job=%s failed", job_id)
        store.mark_failed(job_id, f"{exc.__class__.__name__}: {exc}")
