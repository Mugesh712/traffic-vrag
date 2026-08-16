"""M14 — FastAPI backend.

    POST /upload            accept a video, return job_id, run the pipeline in the background
    GET  /jobs/{id}/status  per-stage pipeline progress
    POST /query             {job_id, question} -> full retrieval + answer object
    GET  /objects/{gid}     object detail + timeline + crops (job_id scoped, see below)
    GET  /frames/{path}     serve an evidence image

VIDEO_ID == JOB_ID. The upload is saved as `<job_id><ext>` before M1 ever
sees it, so M1's filename-derived video_id is always exactly the job_id --
see pipeline_runner.py's docstring for why that removes a whole class of bug.

WHY /objects IS JOB-SCOPED, DEVIATING FROM THE ROADMAP'S BARE `/objects/{gid}`.
M10's own schema deliberately does not treat global_id as globally unique --
"obj_0001" recurs in every video, and Neo4j's uniqueness constraint is on a
video-scoped composite key for exactly that reason (see schema.cypher). A bare
`/objects/{gid}` route would be ambiguous the moment two videos have been
processed, which defeats the multi-video job model this API already commits
to. `/jobs/{job_id}/objects/{gid}` costs one path segment and stays correct.

WHY /frames VALIDATES ITS PATH. `path` is user-supplied and used to read a
file from disk -- without containment checking this is a directory-traversal
vulnerability (`?path=../../../../etc/passwd`). The resolved path is required
to stay inside the project's data directory.

BACKGROUND EXECUTION. FastAPI's BackgroundTasks runs a sync callable in a
worker thread after the response is sent -- enough for a single-user research
demo. It is not a distributed job queue: a restart loses in-flight jobs, and
there's no cross-process worker pool. Fine for M14's stated scope; a real
multi-tenant deployment would need Celery/RQ instead, which M17's docker
compose deliberately does not attempt to set up.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.job_store import JobStore
from src.api.pipeline_runner import run_pipeline
from src.retrieval.answer_generator import generate_answer
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.schemas import FinalObjectIndex

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(title="Traffic-VRAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: JobStore | None = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(settings.resolve_path(settings.api.db_path))
    return _store


class QueryRequest(BaseModel):
    job_id: str
    question: str


def _require_job(job_id: str):
    job = get_store().get_job(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id}")
    return job


@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks) -> dict:
    store = get_store()
    job = store.create_job(video_path="")  # video_path filled in once we know job_id

    # Saved as <job_id><ext> so M1's filename-derived video_id is the job_id.
    suffix = Path(file.filename or "").suffix or ".mp4"
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{job.job_id}{suffix}"

    size = 0
    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                store.mark_failed(job.job_id, f"Upload exceeded {settings.api.max_upload_mb}MB limit")
                raise HTTPException(413, f"File exceeds {settings.api.max_upload_mb}MB limit")
            out.write(chunk)
    await file.close()

    store.set_video_path(job.job_id, str(dest))
    background_tasks.add_task(run_pipeline, job.job_id, str(dest), settings, store)

    logger.info("upload: job_id=%s file=%s size=%d bytes", job.job_id, file.filename, size)
    return {"job_id": job.job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str) -> dict:
    return _require_job(job_id).to_dict()


@app.post("/query")
def query(request: QueryRequest) -> dict:
    job = _require_job(request.job_id)
    if job.status == "failed":
        raise HTTPException(422, f"Job {request.job_id} failed: {job.error}")
    if job.video_id is None:
        raise HTTPException(409, f"Job {request.job_id} has not ingested a video yet")
    if "index" not in job.stages_completed:
        raise HTTPException(
            409,
            f"Job {request.job_id} has not finished indexing yet "
            f"(stage: {job.current_stage}, {job.progress_percent}% complete)",
        )

    result = hybrid_retrieve(request.question, job.video_id, settings=settings)
    answer = generate_answer(result, settings=settings)

    return {
        "answer": answer.answer,
        "status": answer.status,
        "supporting_object_ids": answer.supporting_object_ids,
        "timestamps": [t.model_dump() for t in answer.timestamps],
        "evidence_frames": answer.evidence_frames,
        "kg_subgraph": answer.kg_subgraph.model_dump(),
        "reasoning_trace": answer.reasoning_trace,
        "unsupported_citations": answer.unsupported_citations,
        "retrieval_warnings": result.warnings,
        "count": result.count,
    }


def _load_final_index(job) -> FinalObjectIndex:
    if job.video_id is None:
        raise HTTPException(409, f"Job {job.job_id} has not ingested a video yet")
    path = settings.resolve_path(settings.paths.outputs_dir) / "global_objects_final" / f"{job.video_id}.json"
    if not path.exists():
        raise HTTPException(409, f"Job {job.job_id} has not finished the confirm stage yet")
    return FinalObjectIndex.model_validate_json(path.read_text())


def _object_summary(obj) -> dict:
    return {
        "global_id": obj.global_id,
        "class": obj.cls,
        "attributes": [a.model_dump() for a in obj.attributes],
    }


@app.get("/jobs/{job_id}/objects")
def object_list(job_id: str) -> list[dict]:
    """All objects for a job, with their confirmed attributes -- the M15
    object explorer's data source. Not in the roadmap's original endpoint
    list, which only specified single-object lookup; added because "browse
    all detected objects" has no other way to enumerate what exists."""
    index = _load_final_index(_require_job(job_id))
    return [_object_summary(o) for o in index.objects]


@app.get("/jobs/{job_id}/objects/{global_id}")
def object_detail(job_id: str, global_id: str) -> dict:
    index = _load_final_index(_require_job(job_id))
    obj = next((o for o in index.objects if o.global_id == global_id), None)
    if obj is None:
        raise HTTPException(404, f"No object {global_id} in job {job_id}")

    return {
        **_object_summary(obj),
        "timeline": [s.model_dump() for s in obj.sightings],
        "best_shot_crops": obj.best_shot_crops,
    }


@app.get("/frames/{path:path}")
def frame(path: str) -> FileResponse:
    data_root = settings.resolve_path(settings.paths.data_dir).resolve()
    resolved = (settings.resolve_path(".") / path).resolve()

    # Containment check against the resolved (symlink-free) path, not the raw
    # string, so "../" segments and symlinks are both caught.
    if data_root not in resolved.parents and resolved != data_root:
        raise HTTPException(400, "Path must be inside the data directory")
    if not resolved.is_file():
        raise HTTPException(404, f"No file at {path}")

    return FileResponse(resolved)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
