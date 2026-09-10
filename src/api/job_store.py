"""M14 -- job store.

One record per uploaded video, tracking pipeline progress stage by stage.

Two interchangeable backends, chosen by `api.jobs_backend`:

  "sqlite"    -- a single file at `api.db_path`. The default; needs nothing
                 external. A short-lived connection per operation, because
                 sqlite3 connections are not safe to share across threads and
                 FastAPI's BackgroundTasks run on a worker thread separate
                 from the request that created the job.

  "dynamodb"  -- one item per job in `api.jobs_table_name`, using the shared
                 `aws:` client settings. Credentials come from boto3's normal
                 chain (AWS_* env vars, ~/.aws, an instance role) -- never from
                 config. The table is created on first use if the credentials
                 allow it.

The mutation logic (which fields each transition writes, idempotent stage
accumulation, the updated_at bump) lives once in `_BaseJobStore`; a backend
only implements three storage primitives: `_insert`, `get_job`, `_write`.

`stages_completed` is a list. SQLite has no array type so that backend stores
it as a JSON string and (de)serializes at its boundary; DynamoDB stores it as
a native list.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Fixed, ordered stage list. Per-clip stages (detect..vote) still count as one
# unit each here: progress is reported per pipeline stage, not per clip, which
# is what M14's spec asks for ("pipeline progress per stage").
STAGES: tuple[str, ...] = (
    "ingest", "detect", "track", "associate", "attribute", "vote",
    "link", "confirm", "events", "build_kg", "index",
)

_STATUSES = ("queued", "running", "completed", "failed")

# Every optional column/attribute, so a partial record round-trips with the
# missing ones coming back as None rather than KeyError.
_OPTIONAL_FIELDS = ("video_id", "current_stage", "stage_detail", "error")


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
    def from_record(cls, record: dict) -> "Job":
        """Build a Job from a backend record: a plain dict whose
        `stages_completed` is already a real list and whose optional fields
        may be absent."""
        return cls(
            job_id=record["job_id"],
            video_path=record["video_path"],
            status=record["status"],
            video_id=record.get("video_id"),
            current_stage=record.get("current_stage"),
            stage_detail=record.get("stage_detail"),
            stages_completed=list(record.get("stages_completed") or []),
            error=record.get("error"),
            created_at=record.get("created_at", ""),
            updated_at=record.get("updated_at", ""),
        )


class JobStoreError(RuntimeError):
    pass


@runtime_checkable
class JobStore(Protocol):
    """The contract both backends satisfy and the API layer depends on."""

    def create_job(self, video_path: str, job_id: str | None = None) -> Job: ...
    def get_job(self, job_id: str) -> Job | None: ...
    def set_video_id(self, job_id: str, video_id: str) -> None: ...
    def set_video_path(self, job_id: str, video_path: str) -> None: ...
    def mark_stage_started(self, job_id: str, stage: str, detail: str | None = None) -> None: ...
    def mark_stage_completed(self, job_id: str, stage: str, detail: str | None = None) -> None: ...
    def mark_completed(self, job_id: str) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...


class _BaseJobStore:
    """Shared state transitions. Subclasses implement `_insert`, `get_job`
    and `_write` and inherit every public mutator."""

    def create_job(self, video_path: str, job_id: str | None = None) -> Job:
        job = Job(
            job_id=job_id or str(uuid.uuid4()),
            video_path=video_path,
            status="queued",
            created_at=_now(),
            updated_at=_now(),
        )
        self._insert(job)
        return job

    def set_video_id(self, job_id: str, video_id: str) -> None:
        self._patch(job_id, video_id=video_id)

    def set_video_path(self, job_id: str, video_path: str) -> None:
        self._patch(job_id, video_path=video_path)

    def mark_stage_started(self, job_id: str, stage: str, detail: str | None = None) -> None:
        self._patch(job_id, status="running", current_stage=stage, stage_detail=detail)

    def mark_stage_completed(self, job_id: str, stage: str, detail: str | None = None) -> None:
        job = self.get_job(job_id)
        if job is None:
            raise JobStoreError(f"No job with job_id={job_id}")
        # Per-clip stages call this once after their clip loop, but stay
        # idempotent anyway: a stage is recorded at most once.
        completed = (
            job.stages_completed
            if stage in job.stages_completed
            else job.stages_completed + [stage]
        )
        self._patch(job_id, stages_completed=completed, stage_detail=detail)

    def mark_completed(self, job_id: str) -> None:
        self._patch(job_id, status="completed", current_stage=None, stage_detail=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._patch(job_id, status="failed", error=error)

    def _patch(self, job_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        self._write(job_id, fields)

    # --- storage primitives -------------------------------------------------
    def _insert(self, job: Job) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_job(self, job_id: str) -> Job | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _write(self, job_id: str, fields: dict) -> None:  # pragma: no cover - abstract
        """Apply a partial update. Must raise JobStoreError if job_id is
        absent -- callers rely on that for the unknown-job error path."""
        raise NotImplementedError


class SqliteJobStore(_BaseJobStore):
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
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

    def _insert(self, job: Job) -> None:
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

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["stages_completed"] = json.loads(record["stages_completed"])
        return Job.from_record(record)

    def _write(self, job_id: str, fields: dict) -> None:
        fields = dict(fields)
        if "stages_completed" in fields:
            fields["stages_completed"] = json.dumps(fields["stages_completed"])
        columns = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        with self._connect() as conn:
            cursor = conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", values)
            if cursor.rowcount == 0:
                raise JobStoreError(f"No job with job_id={job_id}")


class DynamoDbJobStore(_BaseJobStore):
    """One item per job, partition key `job_id`. None-valued fields are stored
    as absent attributes (and removed on update), so a record round-trips
    through `Job.from_record` unchanged.

    One instance is shared between the request thread and the BackgroundTasks
    worker thread. That is safe here: every operation is a single put/get/
    update call, which goes through boto3's underlying (thread-safe) client;
    the only resource-level state touched is `_table.load()` in `_ensure_table`,
    which runs once at construction on a single thread.
    """

    def __init__(
        self,
        table_name: str,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        create_if_missing: bool = True,
    ):
        import boto3  # lazy: the sqlite path must not need boto3 installed cleanly

        self.table_name = table_name
        kwargs = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self._dynamodb = boto3.resource("dynamodb", **kwargs)
        self._table = self._dynamodb.Table(table_name)
        if create_if_missing:
            self._ensure_table()

    def _ensure_table(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self._table.load()
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "ResourceNotFoundException":
                # AccessDenied on DescribeTable, throttling, etc. -- assume the
                # table exists and let the first real call surface any problem.
                logger.warning("DynamoDbJobStore: could not describe %s (%s); assuming it exists",
                               self.table_name, code)
                return
        logger.info("DynamoDbJobStore: creating table %s", self.table_name)
        try:
            self._dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
        except ClientError as exc:
            # Another process (e.g. the CLI alongside the API) created it
            # between our describe and our create -- fine, just wait for it.
            if exc.response.get("Error", {}).get("Code") != "ResourceInUseException":
                raise
        self._table.wait_until_exists()

    @staticmethod
    def _item(job: Job) -> dict:
        item = {
            "job_id": job.job_id,
            "video_path": job.video_path,
            "status": job.status,
            "stages_completed": list(job.stages_completed),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        for name in _OPTIONAL_FIELDS:
            value = getattr(job, name)
            if value is not None:
                item[name] = value
        return item

    def _insert(self, job: Job) -> None:
        from botocore.exceptions import ClientError

        try:
            self._table.put_item(
                Item=self._item(job),
                ConditionExpression="attribute_not_exists(job_id)",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise JobStoreError(f"Job {job.job_id} already exists") from exc
            raise

    def get_job(self, job_id: str) -> Job | None:
        item = self._table.get_item(Key={"job_id": job_id}).get("Item")
        return Job.from_record(item) if item else None

    def _write(self, job_id: str, fields: dict) -> None:
        from botocore.exceptions import ClientError

        set_parts, remove_parts = [], []
        names: dict[str, str] = {}
        values: dict[str, object] = {}
        for i, (key, value) in enumerate(fields.items()):
            placeholder = f"#f{i}"
            names[placeholder] = key
            # None means "clear this attribute" (e.g. mark_completed clears
            # current_stage); DynamoDB models that as REMOVE, not SET NULL.
            if value is None:
                remove_parts.append(placeholder)
            else:
                set_parts.append(f"{placeholder} = :v{i}")
                values[f":v{i}"] = value

        expr = ""
        if set_parts:
            expr += "SET " + ", ".join(set_parts)
        if remove_parts:
            expr += (" " if expr else "") + "REMOVE " + ", ".join(remove_parts)

        kwargs = {
            "Key": {"job_id": job_id},
            "UpdateExpression": expr,
            "ExpressionAttributeNames": names,
            "ConditionExpression": "attribute_exists(job_id)",
        }
        if values:
            kwargs["ExpressionAttributeValues"] = values

        try:
            self._table.update_item(**kwargs)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise JobStoreError(f"No job with job_id={job_id}") from exc
            raise


def build_job_store(settings) -> JobStore:
    """Construct the job store named by `settings.api.jobs_backend`."""
    backend = settings.api.jobs_backend.strip().lower()
    if backend == "sqlite":
        return SqliteJobStore(settings.resolve_path(settings.api.db_path))
    if backend == "dynamodb":
        return DynamoDbJobStore(
            settings.api.jobs_table_name,
            region=settings.aws.region or None,
            endpoint_url=settings.aws.endpoint_url or None,
        )
    raise JobStoreError(
        f"Unknown api.jobs_backend {settings.api.jobs_backend!r} (expected 'sqlite' or 'dynamodb')"
    )
