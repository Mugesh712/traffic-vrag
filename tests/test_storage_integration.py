"""End-to-end check that a real pipeline stage works with storage.backend ==
"s3": M1 (video ingest) writes its sampled frames and manifest to S3 (mocked
with moto), and they can be read back through the same abstractions the later
stages use -- image_io.imread and manifest.load_clip_frame_index.

This is the test that would fail if the artifact layer only *looked* right for
local and silently broke on S3.
"""
from __future__ import annotations

import shutil

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from src.ingest.video_ingest import ingest_video
from src.utils import image_io
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.manifest import load_clip_frame_index

BUCKET = "test-tv-artifacts"
REGION = "us-east-1"
PREFIX = "tv"


def _write_video(path, n_frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()


@pytest.fixture
def s3_settings(tmp_path, monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(var, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    settings = get_settings()
    monkeypatch.setattr(settings.storage, "backend", "s3")
    monkeypatch.setattr(settings.storage, "s3_bucket", BUCKET)
    monkeypatch.setattr(settings.storage, "s3_prefix", PREFIX)
    monkeypatch.setattr(settings.aws, "region", REGION)
    # Clip .mp4s are local by design; keep them inside the repo tree (so the
    # manifest's relative_to() still works) but in a throwaway dir we delete.
    clips_dir = PROJECT_ROOT / "data" / "clips_s3_itest"
    monkeypatch.setattr(settings.paths, "clips_dir", str(clips_dir))

    saved = settings._artifact_store
    object.__setattr__(settings, "_artifact_store", None)
    try:
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            yield settings
    finally:
        object.__setattr__(settings, "_artifact_store", saved)
        shutil.rmtree(clips_dir, ignore_errors=True)


def test_ingest_writes_frames_and_manifest_to_s3(s3_settings, tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=20)

    manifest = ingest_video(
        video, settings=s3_settings,
        clip_length_sec=100, frame_sample_interval_sec=0.2,
    )
    video_id = manifest.video_id

    client = boto3.client("s3", region_name=REGION)
    keys = [o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]

    # frames + manifest landed in the bucket, under the configured prefix
    frame_keys = [k for k in keys if k.startswith(f"{PREFIX}/data/frames/{video_id}/")]
    assert frame_keys, keys
    assert f"{PREFIX}/data/outputs/ingest/{video_id}_manifest.json" in keys
    # clip .mp4 did NOT go to S3 (local by design)
    assert not any(k.endswith(".mp4") for k in keys)

    # the manifest is readable back through the artifact store
    manifest_path = s3_settings.resolve_path("data/outputs/ingest") / f"{video_id}_manifest.json"
    assert manifest_path.exists()

    # a recorded frame decodes back to an image via the same helper M3/M5 use
    first_frame_rel = manifest.clips[0].frames[0].frame_path
    img = image_io.imread(s3_settings.resolve_path(first_frame_rel))
    assert img is not None and img.shape == (32, 32, 3)


def test_load_clip_frame_index_resolves_against_s3(s3_settings, tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=20)
    manifest = ingest_video(
        video, settings=s3_settings,
        clip_length_sec=100, frame_sample_interval_sec=0.2,
    )
    clip_id = manifest.clips[0].clip_id

    frame_paths, _, _ = load_clip_frame_index(clip_id, manifest.video_id, settings=s3_settings)

    assert frame_paths
    assert all(v.startswith("data/frames/") for v in frame_paths.values())
