"""Tests for the artifact storage layer (src/utils/storage.py).

The path-arithmetic and IO contract is exercised against BOTH backends via the
parametrized `root` fixture -- the local filesystem and S3 (mocked in-process
with moto). Backend-specific behaviour (os.fspath, mkdir semantics, the
factory) is tested separately.
"""
from __future__ import annotations

import os

import boto3
import pytest
from moto import mock_aws

from src.utils.storage import (
    ArtifactPath,
    LocalStorage,
    S3Storage,
    build_artifact_store,
    build_storage,
)

BUCKET = "test-traffic-vrag"
REGION = "us-east-1"
PREFIX = "traffic-vrag"


@pytest.fixture
def _aws_credentials(monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(var, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture(params=["local", "s3"])
def root(request, tmp_path, _aws_credentials):
    if request.param == "local":
        yield ArtifactPath(LocalStorage(tmp_path), "")
        return
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        yield ArtifactPath(S3Storage(BUCKET, PREFIX, region=REGION), "")


# --------------------------------------------------------------------------
# Path arithmetic -- pure, no backend calls
# --------------------------------------------------------------------------
def test_truediv_builds_nested_keys(root):
    p = root / "data" / "outputs" / "detections" / "clip_000.json"
    assert p.key == "data/outputs/detections/clip_000.json"
    assert p.name == "clip_000.json"
    assert p.stem == "clip_000"
    assert p.suffix == ".json"


def test_parent_walks_up(root):
    p = root / "data" / "outputs" / "x.json"
    assert p.parent.key == "data/outputs"
    assert p.parent.parent.key == "data"
    assert p.parent.parent.parent.key == ""  # the store root


def test_with_name_replaces_the_final_segment(root):
    p = root / "data" / "outputs" / "ingest" / "vid_manifest.json"
    assert p.with_name("vid_other.json").key == "data/outputs/ingest/vid_other.json"


def test_relative_to_returns_the_tail(root):
    p = root / "data" / "frames" / "vid" / "clip_000" / "frame_000001.jpg"
    assert str(p.relative_to(root)) == "data/frames/vid/clip_000/frame_000001.jpg"
    assert str(p.relative_to(root / "data" / "frames")) == "vid/clip_000/frame_000001.jpg"


def test_relative_to_rejects_a_non_ancestor(root):
    p = root / "data" / "outputs" / "x.json"
    with pytest.raises(ValueError):
        p.relative_to(root / "data" / "frames")


def test_dotdot_in_a_key_is_rejected(root):
    with pytest.raises(ValueError):
        root / "data" / ".." / ".." / "etc" / "passwd"


def test_equality_and_hash_track_backend_and_key(root):
    a = root / "data" / "x.json"
    b = root / "data" / "x.json"
    assert a == b and hash(a) == hash(b)
    assert a != root / "data" / "y.json"


def test_resolve_is_identity(root):
    p = root / "data" / "x.json"
    assert p.resolve() == p


# --------------------------------------------------------------------------
# IO contract -- both backends
# --------------------------------------------------------------------------
def test_write_then_read_text_round_trips(root):
    p = root / "data" / "outputs" / "events" / "vid.json"
    assert not p.exists()
    p.write_text('{"events": []}')
    assert p.exists()
    assert p.read_text() == '{"events": []}'


def test_write_then_read_bytes_round_trips(root):
    p = root / "data" / "crops" / "clip_000" / "0.jpg"
    p.write_bytes(b"\xff\xd8\xff\x00binary")
    assert p.read_bytes() == b"\xff\xd8\xff\x00binary"


def test_open_for_write_then_read(root):
    p = root / "data" / "raw" / "upload.bin"
    with p.open("wb") as fh:
        fh.write(b"chunk-1")
        fh.write(b"chunk-2")
    assert p.read_bytes() == b"chunk-1chunk-2"
    with p.open("rb") as fh:
        assert fh.read() == b"chunk-1chunk-2"


def test_reading_a_missing_artifact_raises_filenotfound(root):
    with pytest.raises(FileNotFoundError):
        (root / "data" / "nope.json").read_text()


def test_unlink_removes_and_missing_ok_is_quiet(root):
    p = root / "data" / "outputs" / "tmp.json"
    p.write_text("x")
    p.unlink()
    assert not p.exists()
    p.unlink(missing_ok=True)  # no error


def test_glob_is_non_recursive_and_matches_by_name(root):
    base = root / "data" / "outputs" / "detections"
    (base / "clip_000.json").write_text("a")
    (base / "clip_001.json").write_text("b")
    (base / "notes.txt").write_text("c")
    (base / "sub" / "clip_002.json").write_text("d")  # deeper -- must be excluded

    hits = sorted(p.name for p in base.glob("*.json"))
    assert hits == ["clip_000.json", "clip_001.json"]


# --------------------------------------------------------------------------
# Backend-specific
# --------------------------------------------------------------------------
def test_local_fspath_is_a_real_filesystem_path(tmp_path):
    p = ArtifactPath(LocalStorage(tmp_path), "") / "data" / "x.json"
    assert os.fspath(p) == str(tmp_path / "data" / "x.json")
    # usable anywhere a path string is expected
    p.write_text("hi")
    with open(p) as fh:
        assert fh.read() == "hi"


def test_s3_fspath_raises_because_there_is_no_local_file(_aws_credentials):
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        p = ArtifactPath(S3Storage(BUCKET, PREFIX, region=REGION), "") / "data" / "x.json"
        with pytest.raises(TypeError):
            os.fspath(p)
        assert str(p) == f"s3://{BUCKET}/{PREFIX}/data/x.json"


def test_local_mkdir_creates_the_directory(tmp_path):
    d = ArtifactPath(LocalStorage(tmp_path), "") / "data" / "outputs" / "new"
    d.mkdir(parents=True, exist_ok=True)
    assert (tmp_path / "data" / "outputs" / "new").is_dir()


def test_s3_mkdir_is_a_noop(_aws_credentials):
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        d = ArtifactPath(S3Storage(BUCKET, PREFIX, region=REGION), "") / "data" / "outputs"
        d.mkdir(parents=True, exist_ok=True)  # must not raise


def test_s3_prefix_is_applied_to_the_stored_key(_aws_credentials):
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        p = ArtifactPath(S3Storage(BUCKET, PREFIX, region=REGION), "") / "data" / "outputs" / "x.json"
        p.write_text("{}")
        listed = client.list_objects_v2(Bucket=BUCKET)["Contents"]
        assert [o["Key"] for o in listed] == ["traffic-vrag/data/outputs/x.json"]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def _settings(monkeypatch, **storage):
    from src.utils.config import get_settings

    s = get_settings()
    for k, v in storage.items():
        monkeypatch.setattr(s.storage, k, v)
    return s


def test_build_storage_defaults_to_local(monkeypatch):
    s = _settings(monkeypatch, backend="local")
    assert isinstance(build_storage(s), LocalStorage)
    assert isinstance(build_artifact_store(s), ArtifactPath)


def test_build_storage_builds_s3(monkeypatch, _aws_credentials):
    s = _settings(monkeypatch, backend="s3", s3_bucket=BUCKET, s3_prefix=PREFIX)
    monkeypatch.setattr(s.aws, "region", REGION)
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        store = build_storage(s)
        assert isinstance(store, S3Storage)
        root = build_artifact_store(s)
        (root / "data" / "ping.txt").write_text("pong")
        assert (root / "data" / "ping.txt").read_text() == "pong"


def test_build_storage_rejects_unknown_backend(monkeypatch):
    s = _settings(monkeypatch, backend="gcs")
    with pytest.raises(ValueError):
        build_storage(s)


def test_s3_storage_requires_a_bucket():
    with pytest.raises(ValueError):
        S3Storage("", PREFIX)
