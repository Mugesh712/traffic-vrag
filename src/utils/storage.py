"""Artifact storage -- a `pathlib.Path`-like handle to everything the pipeline
writes under `data/`, backed by either the local filesystem or S3.

WHY THIS EXISTS. Every stage hands its output to the next as files on disk
(`data/outputs/<stage>/<id>.json`, frames, crops, the vector store, eval
tables). That is fine on one machine; it breaks the moment stages run on
separate nodes or on ephemeral compute, where there is no shared disk. Routing
those reads and writes through one object lets the same code target S3 without
each stage knowing.

WHAT IS *NOT* AN ARTIFACT. Model weights, the SQLite job database and
ChromaDB's persist directory are infrastructure, not pipeline artifacts --
they stay local always and do not go through here.

SCOPE OF THE FACADE. `ArtifactPath` implements only the operations the
pipeline actually performs on `settings.resolve_path(...)` results: `/`,
`parent`, `name`, `stem`, `suffix`, `with_name`, `read_text` / `write_text`,
`read_bytes` / `write_bytes`, `open`, `exists`, `is_file`, `mkdir`, `unlink`,
`glob`, `relative_to`, `resolve`, and `str()` / `os.fspath()`. It is
deliberately not a general-purpose Path.

NOT WIRED IN YET. `resolve_path()` still returns a real `pathlib.Path`;
migrating the stages onto `ArtifactPath` is a separate, staged change. This
module and `build_artifact_store()` stand alone so the abstraction can be
reviewed before anything depends on it.
"""
from __future__ import annotations

import fnmatch
import io
import os
import posixpath
from pathlib import Path, PurePosixPath
from typing import IO, Iterator, Protocol, runtime_checkable

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize(key: str) -> str:
    """Collapse a key to a clean, root-relative POSIX string.

    "." and "" -> "" (the store root); leading slashes and "./" are stripped;
    ".." is rejected -- an artifact key must never escape the store.
    """
    key = str(key).replace(os.sep, "/").strip()
    if key in ("", "."):
        return ""
    parts: list[str] = []
    for part in key.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"artifact key must not contain '..': {key!r}")
        parts.append(part)
    return "/".join(parts)


@runtime_checkable
class Storage(Protocol):
    """Byte-level backend. Keys are always clean root-relative POSIX strings."""

    root_uri: str

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def iter_keys(self, prefix: str) -> Iterator[str]: ...
    def delete(self, key: str, missing_ok: bool = False) -> None: ...
    def local_path(self, key: str) -> Path | None: ...
    def open(self, key: str, mode: str) -> IO: ...


class LocalStorage:
    """Rooted at a real directory (the project root, in normal use)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root_uri = self.root.as_uri() if self.root.is_absolute() else str(self.root)

    def _abs(self, key: str) -> Path:
        return self.root / key if key else self.root

    def read_bytes(self, key: str) -> bytes:
        return self._abs(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        target = self._abs(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._abs(key).exists()

    def iter_keys(self, prefix: str) -> Iterator[str]:
        base = self._abs(prefix)
        if not base.exists():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path.relative_to(self.root).as_posix()

    def delete(self, key: str, missing_ok: bool = False) -> None:
        self._abs(key).unlink(missing_ok=missing_ok)

    def local_path(self, key: str) -> Path:
        return self._abs(key)

    def open(self, key: str, mode: str) -> IO:
        target = self._abs(key)
        if any(c in mode for c in "wxa"):
            target.parent.mkdir(parents=True, exist_ok=True)
        return open(target, mode)


class _S3Writer(io.BytesIO):
    """A write handle that uploads its buffer to S3 on close()."""

    def __init__(self, storage: "S3Storage", key: str, text: bool):
        super().__init__()
        self._storage = storage
        self._key = key
        self._text = text
        self._done = False

    def close(self) -> None:
        if not self._done:
            self._done = True
            self._storage.write_bytes(self._key, self.getvalue())
        super().close()

    def __exit__(self, *exc) -> None:
        self.close()


class S3Storage:
    """s3://<bucket>/<prefix>/<key>. S3 has no directories, so `mkdir` is a
    no-op at the `ArtifactPath` layer and `iter_keys` is a prefix listing."""

    def __init__(self, bucket: str, prefix: str = "", *, region: str | None = None,
                 endpoint_url: str | None = None, client=None):
        if not bucket:
            raise ValueError("S3Storage needs a bucket name (set storage.s3_bucket)")
        self.bucket = bucket
        self.prefix = _normalize(prefix)
        self.root_uri = f"s3://{bucket}/{self.prefix}".rstrip("/")
        if client is not None:
            self._client = client
        else:
            import boto3  # lazy: the local path must not need boto3

            kwargs = {}
            if region:
                kwargs["region_name"] = region
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("s3", **kwargs)

    def _s3_key(self, key: str) -> str:
        return posixpath.join(self.prefix, key) if self.prefix else key

    def read_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            return self._client.get_object(Bucket=self.bucket, Key=self._s3_key(key))["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"{self.root_uri}/{key}") from exc
            raise

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._s3_key(key), Body=data)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def iter_keys(self, prefix: str) -> Iterator[str]:
        s3_prefix = self._s3_key(_normalize(prefix))
        if s3_prefix and not s3_prefix.endswith("/"):
            s3_prefix += "/"
        paginator = self._client.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"][strip:]

    def delete(self, key: str, missing_ok: bool = False) -> None:
        # S3 delete_object is idempotent -- it never errors on a missing key,
        # so missing_ok is satisfied for free.
        self._client.delete_object(Bucket=self.bucket, Key=self._s3_key(key))

    def local_path(self, key: str) -> None:
        return None

    def open(self, key: str, mode: str) -> IO:
        if any(c in mode for c in "wxa"):
            return _S3Writer(self, key, text="b" not in mode)
        data = self.read_bytes(key)
        return io.BytesIO(data) if "b" in mode else io.StringIO(data.decode())


class ArtifactPath:
    """A path-like handle to one artifact. Cheap to construct and to `/`;
    only the IO methods touch the backend."""

    __slots__ = ("_storage", "_key")

    def __init__(self, storage: Storage, key: str = ""):
        self._storage = storage
        self._key = _normalize(key)

    # --- pure path arithmetic --------------------------------------------
    def _sibling(self, key: str) -> "ArtifactPath":
        return ArtifactPath(self._storage, key)

    def __truediv__(self, other) -> "ArtifactPath":
        other = other._key if isinstance(other, ArtifactPath) else str(other)
        joined = f"{self._key}/{other}" if self._key else other
        return self._sibling(joined)

    @property
    def parent(self) -> "ArtifactPath":
        return self._sibling(self._key.rsplit("/", 1)[0] if "/" in self._key else "")

    @property
    def name(self) -> str:
        return self._key.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix

    def with_name(self, name: str) -> "ArtifactPath":
        parent_key = self._key.rsplit("/", 1)[0] if "/" in self._key else ""
        return self._sibling(f"{parent_key}/{name}" if parent_key else name)

    def relative_to(self, other) -> PurePosixPath:
        other_key = other._key if isinstance(other, ArtifactPath) else _normalize(other)
        if other_key and not (self._key == other_key or self._key.startswith(other_key + "/")):
            raise ValueError(f"{self._key!r} is not relative to {other_key!r}")
        rel = self._key[len(other_key):].lstrip("/") if other_key else self._key
        return PurePosixPath(rel)

    def resolve(self) -> "ArtifactPath":
        return self  # keys are already normalized and absolute within the store

    @property
    def key(self) -> str:
        return self._key

    # --- IO -------------------------------------------------------------
    def read_bytes(self) -> bytes:
        return self._storage.read_bytes(self._key)

    def write_bytes(self, data: bytes) -> None:
        self._storage.write_bytes(self._key, data)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._storage.read_bytes(self._key).decode(encoding)

    def write_text(self, data: str, encoding: str = "utf-8") -> None:
        self._storage.write_bytes(self._key, data.encode(encoding))

    def open(self, mode: str = "r") -> IO:
        return self._storage.open(self._key, mode)

    def exists(self) -> bool:
        return self._storage.exists(self._key)

    def is_file(self) -> bool:
        # No directory concept in S3; locally, defer to the real check.
        local = self._storage.local_path(self._key)
        return local.is_file() if local is not None else self._storage.exists(self._key)

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        local = self._storage.local_path(self._key)
        if local is not None:
            local.mkdir(parents=parents, exist_ok=exist_ok)
        # S3: nothing to do -- keys carry their own prefix.

    def unlink(self, missing_ok: bool = False) -> None:
        self._storage.delete(self._key, missing_ok=missing_ok)

    def glob(self, pattern: str) -> Iterator["ArtifactPath"]:
        """Non-recursive, like `Path.glob('*.json')`: only immediate children
        of this key, matched by name."""
        prefix = f"{self._key}/" if self._key else ""
        for key in self._storage.iter_keys(self._key):
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix):]
            if "/" in rel:
                continue
            if fnmatch.fnmatch(rel, pattern):
                yield self._sibling(key)

    # --- interop ------------------------------------------------------
    def __fspath__(self) -> str:
        local = self._storage.local_path(self._key)
        if local is None:
            raise TypeError(
                f"{self!r} is not a local file; read_bytes()/open() it instead of "
                "passing it to code that expects a filesystem path"
            )
        return os.fspath(local)

    def __str__(self) -> str:
        local = self._storage.local_path(self._key)
        return os.fspath(local) if local is not None else f"{self._storage.root_uri}/{self._key}".rstrip("/")

    def __repr__(self) -> str:
        return f"ArtifactPath({self.__str__()!r})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ArtifactPath)
            and other._storage.root_uri == self._storage.root_uri
            and other._key == self._key
        )

    def __hash__(self) -> int:
        return hash((self._storage.root_uri, self._key))


def build_storage(settings) -> Storage:
    """Construct the byte backend named by `settings.storage.backend`."""
    backend = settings.storage.backend.strip().lower()
    if backend == "local":
        return LocalStorage(PROJECT_ROOT)
    if backend == "s3":
        return S3Storage(
            settings.storage.s3_bucket,
            settings.storage.s3_prefix,
            region=settings.aws.region or None,
            endpoint_url=settings.aws.endpoint_url or None,
        )
    raise ValueError(
        f"Unknown storage.backend {settings.storage.backend!r} (expected 'local' or 's3')"
    )


def build_artifact_store(settings) -> ArtifactPath:
    """The store root as an ArtifactPath -- `build_artifact_store(s) / 'data' / ...`."""
    return ArtifactPath(build_storage(settings), "")
