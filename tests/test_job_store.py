"""Tests for M14's job store.

Every behavioural test runs against BOTH backends via the parametrized `store`
fixture -- the SQLite file store and the DynamoDB store (mocked in-process with
moto). The contract is identical, so the assertions are too; anything
backend-specific (table creation, attribute removal, the factory) is a
separate test below.
"""
from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from src.api.job_store import (
    STAGES,
    DynamoDbJobStore,
    JobStoreError,
    SqliteJobStore,
    build_job_store,
)
from src.utils.config import get_settings

TABLE = "test-traffic-vrag-jobs"
REGION = "us-east-1"


@pytest.fixture
def _aws_credentials(monkeypatch):
    """moto intercepts the API calls, but boto3 still needs *some* creds to
    sign the request and a region to target."""
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(var, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture(params=["sqlite", "dynamodb"])
def store(request, tmp_path, _aws_credentials):
    if request.param == "sqlite":
        yield SqliteJobStore(tmp_path / "jobs.db")
        return
    with mock_aws():
        yield DynamoDbJobStore(TABLE, region=REGION)


# --------------------------------------------------------------------------
# Contract -- runs for sqlite AND dynamodb
# --------------------------------------------------------------------------
def test_create_job_starts_queued_with_no_progress(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    assert job.status == "queued"
    assert job.stages_completed == []
    assert job.progress_percent == 0.0
    assert job.video_id is None


def test_get_job_round_trips(store):
    created = store.create_job(video_path="/tmp/x.mp4")
    fetched = store.get_job(created.job_id)
    assert fetched.job_id == created.job_id
    assert fetched.video_path == "/tmp/x.mp4"
    assert fetched.status == "queued"
    assert fetched.stages_completed == []


def test_get_job_returns_none_for_unknown_id(store):
    assert store.get_job("does-not-exist") is None


def test_mark_stage_started_sets_running_and_current_stage(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_stage_started(job.job_id, "detect", detail="clip 1/3: clip_000")
    fetched = store.get_job(job.job_id)
    assert fetched.status == "running"
    assert fetched.current_stage == "detect"
    assert fetched.stage_detail == "clip 1/3: clip_000"


def test_mark_stage_completed_accumulates_and_updates_progress(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_stage_completed(job.job_id, "ingest")
    store.mark_stage_completed(job.job_id, "detect")
    fetched = store.get_job(job.job_id)
    assert fetched.stages_completed == ["ingest", "detect"]
    assert fetched.progress_percent == pytest.approx(200 / len(STAGES), abs=0.1)


def test_stage_completion_is_idempotent(store):
    """A per-clip stage (detect, track, ...) must be recorded only once even
    if mark_stage_completed is called for it repeatedly."""
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_stage_completed(job.job_id, "detect")
    store.mark_stage_completed(job.job_id, "detect")
    fetched = store.get_job(job.job_id)
    assert fetched.stages_completed == ["detect"]


def test_mark_completed_clears_current_stage(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_stage_started(job.job_id, "index")
    store.mark_completed(job.job_id)
    fetched = store.get_job(job.job_id)
    assert fetched.status == "completed"
    assert fetched.current_stage is None
    assert fetched.stage_detail is None


def test_mark_failed_records_the_error_and_preserves_progress(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_stage_completed(job.job_id, "ingest")
    store.mark_failed(job.job_id, "DetectorError: boom")
    fetched = store.get_job(job.job_id)
    assert fetched.status == "failed"
    assert fetched.error == "DetectorError: boom"
    assert fetched.stages_completed == ["ingest"]  # partial progress is not discarded


def test_set_video_id_and_video_path(store):
    job = store.create_job(video_path="")
    store.set_video_path(job.job_id, "/data/raw/abc.mp4")
    store.set_video_id(job.job_id, "abc")
    fetched = store.get_job(job.job_id)
    assert fetched.video_path == "/data/raw/abc.mp4"
    assert fetched.video_id == "abc"


def test_operations_on_unknown_job_raise(store):
    with pytest.raises(JobStoreError):
        store.mark_stage_completed("no-such-job", "ingest")
    with pytest.raises(JobStoreError):
        store.mark_failed("no-such-job", "boom")


def test_progress_percent_reaches_100_when_all_stages_done(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    for stage in STAGES:
        store.mark_stage_completed(job.job_id, stage)
    fetched = store.get_job(job.job_id)
    assert fetched.progress_percent == 100.0


def test_to_dict_is_json_serializable_shape(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    d = job.to_dict()
    assert set(d) == {
        "job_id", "video_id", "status", "current_stage", "stage_detail",
        "stages_completed", "total_stages", "progress_percent", "error",
        "created_at", "updated_at",
    }


def test_updated_at_moves_forward_on_mutation(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    first = store.get_job(job.job_id).updated_at
    store.mark_stage_started(job.job_id, "ingest")
    assert store.get_job(job.job_id).updated_at >= first


# --------------------------------------------------------------------------
# DynamoDB-specific
# --------------------------------------------------------------------------
def test_dynamodb_creates_the_table_on_first_use(_aws_credentials):
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        assert TABLE not in client.list_tables()["TableNames"]
        DynamoDbJobStore(TABLE, region=REGION)
        assert TABLE in client.list_tables()["TableNames"]


def test_dynamodb_reuses_an_existing_table(_aws_credentials):
    with mock_aws():
        DynamoDbJobStore(TABLE, region=REGION)
        # A second construction must not fail trying to re-create it.
        store = DynamoDbJobStore(TABLE, region=REGION)
        job = store.create_job(video_path="/tmp/x.mp4")
        assert store.get_job(job.job_id).job_id == job.job_id


def test_dynamodb_clears_a_field_by_removing_the_attribute(_aws_credentials):
    """None means REMOVE, not a stored NULL -- so a cleared field is simply
    absent from the item and comes back as None."""
    with mock_aws():
        store = DynamoDbJobStore(TABLE, region=REGION)
        job = store.create_job(video_path="/tmp/x.mp4")
        store.mark_stage_started(job.job_id, "index", detail="clip 1/1")
        store.mark_completed(job.job_id)

        raw = boto3.resource("dynamodb", region_name=REGION).Table(TABLE).get_item(
            Key={"job_id": job.job_id}
        )["Item"]
        assert "current_stage" not in raw
        assert "stage_detail" not in raw
        assert store.get_job(job.job_id).current_stage is None


def test_dynamodb_rejects_a_duplicate_job_id(_aws_credentials):
    with mock_aws():
        store = DynamoDbJobStore(TABLE, region=REGION)
        job = store.create_job(video_path="/tmp/x.mp4")
        with pytest.raises(JobStoreError):
            store.create_job(video_path="/tmp/y.mp4", job_id=job.job_id)


def test_dynamodb_stores_stages_completed_as_a_native_list(_aws_credentials):
    with mock_aws():
        store = DynamoDbJobStore(TABLE, region=REGION)
        job = store.create_job(video_path="/tmp/x.mp4")
        store.mark_stage_completed(job.job_id, "ingest")
        store.mark_stage_completed(job.job_id, "detect")
        raw = boto3.resource("dynamodb", region_name=REGION).Table(TABLE).get_item(
            Key={"job_id": job.job_id}
        )["Item"]
        assert raw["stages_completed"] == ["ingest", "detect"]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def test_build_job_store_defaults_to_sqlite(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.api, "jobs_backend", "sqlite")
    monkeypatch.setattr(settings.api, "db_path", str(tmp_path / "jobs.db"))
    assert isinstance(build_job_store(settings), SqliteJobStore)


def test_build_job_store_builds_dynamodb(monkeypatch, _aws_credentials):
    settings = get_settings()
    monkeypatch.setattr(settings.api, "jobs_backend", "dynamodb")
    monkeypatch.setattr(settings.api, "jobs_table_name", TABLE)
    monkeypatch.setattr(settings.aws, "region", REGION)
    with mock_aws():
        store = build_job_store(settings)
        assert isinstance(store, DynamoDbJobStore)
        job = store.create_job(video_path="/tmp/x.mp4")
        assert store.get_job(job.job_id).job_id == job.job_id


def test_build_job_store_rejects_an_unknown_backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.api, "jobs_backend", "postgres")
    with pytest.raises(JobStoreError):
        build_job_store(settings)
