"""Tests for M14's FastAPI endpoints, against the real app via TestClient.

The heavy pipeline (run_pipeline) is replaced with a fast fake that exercises
the same job-store contract, so these tests validate the HTTP layer -- status
codes, request/response shapes, path-traversal guarding -- without running
YOLO/Florence-2/Neo4j. Fixtures live under the real data/ tree (same
constraint as every other module's live tests) and are cleaned up.
"""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.job_store import JobStore
from src.utils.config import get_settings

VIDEO_ID = "video_test_m14"


def fake_run_pipeline(job_id, video_path, settings, store):
    """Stands in for the real pipeline: ingest sets video_id=job_id (as the
    real M1 stage would, given uploads are saved as <job_id><ext>), then every
    stage completes immediately."""
    from src.api.job_store import STAGES

    store.set_video_id(job_id, job_id)
    for stage in STAGES:
        store.mark_stage_completed(job_id, stage)
    store.mark_completed(job_id)


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.api, "db_path", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(api_main, "_store", None)
    monkeypatch.setattr(api_main, "run_pipeline", fake_run_pipeline)
    with TestClient(api_main.app) as c:
        yield c

    outputs = settings.resolve_path(settings.paths.outputs_dir)
    (outputs / "global_objects_final" / f"{VIDEO_ID}.json").unlink(missing_ok=True)
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    for f in raw_dir.glob(f"{VIDEO_ID}.*"):
        f.unlink()


def upload_fixture_video(client) -> str:
    """Upload with a controlled filename so the resulting job_id is
    predictable-ish is not guaranteed (job_id is a uuid) -- instead we read
    job_id back from the response, matching how a real client would."""
    video_bytes = b"not a real video, just bytes for the upload path"
    response = client.post(
        "/upload", files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")}
    )
    assert response.status_code == 200
    return response.json()["job_id"]


# --- /upload -----------------------------------------------------------


def test_upload_returns_a_job_id_and_saves_the_file(client):
    job_id = upload_fixture_video(client)
    settings = get_settings()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    assert (raw_dir / f"{job_id}.mp4").exists()
    (raw_dir / f"{job_id}.mp4").unlink()


def test_upload_runs_the_pipeline_in_the_background_and_completes(client):
    """With TestClient, BackgroundTasks run before the response context exits,
    so by the time we check status the fake pipeline has already finished."""
    job_id = upload_fixture_video(client)
    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["status"] == "completed"
    assert status["progress_percent"] == 100.0
    settings = get_settings()
    (settings.resolve_path(settings.paths.raw_dir) / f"{job_id}.mp4").unlink(missing_ok=True)


# --- /jobs/{id}/status ---------------------------------------------------


def test_status_for_unknown_job_is_404(client):
    response = client.get("/jobs/does-not-exist/status")
    assert response.status_code == 404


# --- /query --------------------------------------------------------------


def test_query_before_indexing_finishes_is_409(client, monkeypatch):
    settings = get_settings()
    store = api_main.get_store()
    job = store.create_job(video_path="/tmp/x.mp4")
    store.set_video_id(job.job_id, job.job_id)
    store.mark_stage_completed(job.job_id, "ingest")  # far short of "index"

    response = client.post("/query", json={"job_id": job.job_id, "question": "what happened?"})
    assert response.status_code == 409


def test_query_for_unknown_job_is_404(client):
    response = client.post("/query", json={"job_id": "no-such-job", "question": "x"})
    assert response.status_code == 404


def test_query_for_failed_job_is_422(client):
    store = api_main.get_store()
    job = store.create_job(video_path="/tmp/x.mp4")
    store.mark_failed(job.job_id, "DetectorError: boom")

    response = client.post("/query", json={"job_id": job.job_id, "question": "x"})
    assert response.status_code == 422
    assert "boom" in response.json()["detail"]


def test_query_after_completion_calls_retrieval_and_answer_generation(client, monkeypatch):
    """Full contract of a successful query, with M12/M13 themselves mocked
    (they have their own live-service tests) -- this checks the API wires
    the two together and shapes the response correctly."""
    from src.utils.schemas import AnswerResult, QueryIntent, RetrievalResult

    job_id = upload_fixture_video(client)

    def fake_retrieve(question, video_id, settings=None):
        assert video_id == job_id
        return RetrievalResult(question=question, video_id=video_id, intent=QueryIntent(question=question))

    def fake_answer(result, settings=None):
        return AnswerResult(answer="A white car overtook a truck.", status="answered", supporting_object_ids=["obj_1"])

    monkeypatch.setattr(api_main, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(api_main, "generate_answer", fake_answer)

    response = client.post("/query", json={"job_id": job_id, "question": "who overtook?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A white car overtook a truck."
    assert body["supporting_object_ids"] == ["obj_1"]

    settings = get_settings()
    (settings.resolve_path(settings.paths.raw_dir) / f"{job_id}.mp4").unlink(missing_ok=True)


# --- /jobs/{job_id}/objects/{gid} ----------------------------------------


def test_object_detail_before_confirm_stage_is_409(client):
    store = api_main.get_store()
    job = store.create_job(video_path="/tmp/x.mp4")
    store.set_video_id(job.job_id, "video_with_no_output_file")

    response = client.get(f"/jobs/{job.job_id}/objects/obj_0001")
    assert response.status_code == 409


def test_object_detail_returns_the_matching_object(client):
    settings = get_settings()
    store = api_main.get_store()
    job = store.create_job(video_path="/tmp/x.mp4")
    store.set_video_id(job.job_id, VIDEO_ID)

    outputs = settings.resolve_path(settings.paths.outputs_dir)
    (outputs / "global_objects_final").mkdir(parents=True, exist_ok=True)
    (outputs / "global_objects_final" / f"{VIDEO_ID}.json").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "objects": [
                    {
                        "global_id": "obj_0001", "class": "car",
                        "sightings": [{"clip_id": "c1", "track_id": "1",
                                       "first_seen": "2026-08-15T11:00:00", "last_seen": "2026-08-15T11:00:30",
                                       "gate_scores": {}}],
                        "attributes": [{"attribute": "color", "value": "white", "confidence": 0.9,
                                        "source": "agreed", "uncertain": False}],
                        "best_shot_crops": ["data/crops/c1/1/f0.jpg"],
                    }
                ],
                "correction_log": [],
            }
        )
    )

    response = client.get(f"/jobs/{job.job_id}/objects/obj_0001")
    assert response.status_code == 200
    body = response.json()
    assert body["global_id"] == "obj_0001"
    assert body["attributes"][0]["value"] == "white"
    assert body["best_shot_crops"] == ["data/crops/c1/1/f0.jpg"]


def test_object_detail_for_unknown_global_id_is_404(client):
    settings = get_settings()
    store = api_main.get_store()
    job = store.create_job(video_path="/tmp/x.mp4")
    store.set_video_id(job.job_id, VIDEO_ID)
    outputs = settings.resolve_path(settings.paths.outputs_dir)
    (outputs / "global_objects_final").mkdir(parents=True, exist_ok=True)
    (outputs / "global_objects_final" / f"{VIDEO_ID}.json").write_text(
        json.dumps({"video_id": VIDEO_ID, "objects": [], "correction_log": []})
    )

    response = client.get(f"/jobs/{job.job_id}/objects/obj_9999")
    assert response.status_code == 404


# --- /frames/{path} -- path traversal ------------------------------------


def test_frame_serves_a_real_file_inside_the_data_directory(client, tmp_path):
    settings = get_settings()
    frames_dir = settings.resolve_path(settings.paths.frames_dir) / "clip_test_m14"
    frames_dir.mkdir(parents=True, exist_ok=True)
    target = frames_dir / "frame_000000.jpg"
    target.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    response = client.get("/frames/data/frames/clip_test_m14/frame_000000.jpg")
    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff\xe0fakejpegbytes"
    target.unlink()
    frames_dir.rmdir()


def test_frame_rejects_path_traversal_outside_the_data_directory(client):
    """The core security check: '../' segments escaping data/ must be
    rejected, not silently resolved and served."""
    response = client.get("/frames/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)  # never 200


def test_frame_rejects_absolute_path_escape(client):
    response = client.get("/frames/../../../../../../etc/passwd")
    assert response.status_code in (400, 404)


def test_frame_404s_for_missing_file(client):
    response = client.get("/frames/data/frames/does_not_exist/frame.jpg")
    assert response.status_code == 404


# --- /health ---------------------------------------------------------------


def test_health_check(client):
    assert client.get("/health").json() == {"status": "ok"}
