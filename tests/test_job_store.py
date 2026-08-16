"""Tests for M14's SQLite job store."""
from __future__ import annotations

import pytest

from src.api.job_store import STAGES, JobStore, JobStoreError


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def test_create_job_starts_queued_with_no_progress(store):
    job = store.create_job(video_path="/tmp/x.mp4")
    assert job.status == "queued"
    assert job.stages_completed == []
    assert job.progress_percent == 0.0
    assert job.video_id is None


def test_get_job_round_trips_through_sqlite(store):
    created = store.create_job(video_path="/tmp/x.mp4")
    fetched = store.get_job(created.job_id)
    assert fetched.job_id == created.job_id
    assert fetched.video_path == "/tmp/x.mp4"


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
    """A stage that runs per-clip (detect, track, ...) must only be recorded
    once in stages_completed -- the pipeline runner calls mark_stage_completed
    once per stage after its clip loop, not once per clip, but this must hold
    even if called multiple times."""
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
