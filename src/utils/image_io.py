"""OpenCV image read/write that also works when the path is an S3-backed
`ArtifactPath`.

`cv2.imread` / `cv2.imwrite` only understand local filesystem paths. These
wrappers keep the fast local path (hand the real filename straight to OpenCV)
and fall back to encode/decode-in-memory plus a bytes transfer when the
artifact lives on S3.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.storage import ArtifactPath


def imread(path: ArtifactPath | Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Like cv2.imread: returns the image, or None if it could not be read."""
    if isinstance(path, ArtifactPath):
        local = path.local_path
        if local is not None:
            return cv2.imread(str(local), flags)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    return cv2.imread(str(path), flags)


def imwrite(path: ArtifactPath | Path | str, image: np.ndarray) -> None:
    """Like cv2.imwrite. The encoding is chosen from the path's suffix
    (default .jpg). Raises on an encode failure rather than returning False,
    since a silently dropped frame is a real defect here."""
    if isinstance(path, ArtifactPath):
        local = path.local_path
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(local), image):
                raise OSError(f"cv2.imwrite failed for {local}")
            return
        ext = path.suffix or ".jpg"
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            raise OSError(f"cv2.imencode failed for {path} (ext {ext!r})")
        path.write_bytes(encoded.tobytes())
        return
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cv2.imwrite failed for {path}")
