"""M14 — SQLite job store.

One row per uploaded video. A short-lived connection is opened per operation
rather than shared across requests/threads, since sqlite3 connections are not
safe to share across threads and FastAPI's BackgroundTasks run on a worker
thread separate from the request that created the job.

`stages_completed` has no native SQLite array type, so it is stored as a JSON
string and (de)serialized at the boundary.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Fixed, ordered stage list. Per-clip stages (detect..vote) still count as one
# unit each here: progress is reported per pipeline stage, not per clip, which
# is what M14's spec asks for ("pipeline progress per stage").
STAGES: tuple[str, ...] = (
    "ingest", "detect", "track", "associate", "attribute", "vote",
    "link", "confirm", "events", "build_kg", "index",
)

_STATUSES = ("queued", "running", "completed", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    video_path: str
    status: str
    video_id: str | None = None
    current_stage: str | None = None
    stage_detail: str | None = None
    stages_completed: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def progress_percent(self) -> float:
        return round(100 * len(self.stages_completed) / len(STAGES), 1)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stage_detail": self.stage_detail,
            "stages_completed": self.stages_completed,
            "total_stages": len(STAGES),
            "progress_percent": self.progress_percent,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            job_id=row["job_id"],
            video_path=row["video_path"],
            status=row["status"],
            video_id=row["video_id"],
            current_stage=row["current_stage"],
            stage_detail=row["stage_detail"],
            stages_completed=json.loads(row["stages_completed"]),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class JobStoreError(RuntimeError):
    pass


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_id TEXT,
                    video_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    stage_detail TEXT,
                    stages_completed TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_job(self, video_path: str, job_id: str | None = None) -> Job:
        job = Job(
            job_id=job_id or str(uuid.uuid4()),
            video_path=video_path,
            status="queued",
            created_at=_now(),
            updated_at=_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (job_id, video_id, video_path, status, current_stage, stage_detail,
                    stages_completed, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job.job_id, job.video_id, job.video_path, job.status, job.current_stage,
                 job.stage_detail, json.dumps(job.stages_completed), job.error,
                 job.created_at, job.updated_at),
            )
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return Job._from_row(row) if row else None

    def _update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        columns = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        with self._connect() as conn:
            cursor = conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", values)
            if cursor.rowcount == 0:
                raise JobStoreError(f"No job with job_id={job_id}")

    def set_video_id(self, job_id: str, video_id: str) -> None:
        self._update(job_id, video_id=video_id)

    def set_video_path(self, job_id: str, video_path: str) -> None:
        self._update(job_id, video_path=video_path)

    def mark_stage_started(self, job_id: str, stage: str, detail: str | None = None) -> None:
        self._update(job_id, status="running", current_stage=stage, stage_detail=detail)

    def mark_stage_completed(self, job_id: str, stage: str, detail: str | None = None) -> None:
        job = self.get_job(job_id)
        if job is None:
            raise JobStoreError(f"No job with job_id={job_id}")
        completed = job.stages_completed + [stage] if stage not in job.stages_completed else job.stages_completed
        self._update(job_id, stages_completed=json.dumps(completed), stage_detail=detail)

    def mark_completed(self, job_id: str) -> None:
        self._update(job_id, status="completed", current_stage=None, stage_detail=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", error=error)
