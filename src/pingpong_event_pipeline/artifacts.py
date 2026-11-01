"""Content-addressed review artifacts with atomic replacement.

Numerical results and video pixels must describe the same inputs. A file's
existence or frame count alone is insufficient evidence that it can be reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


@contextmanager
def atomic_path(destination):
    """Keep the previous artifact until its replacement has been validated."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.stem}-",
                                     dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        yield staged
        if not staged.is_file():
            raise FileNotFoundError(f"Artifact producer did not create {staged}")
        staged.replace(destination)


def atomic_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_path(path) as staged:
        staged.write_text(content, encoding="utf-8")


def validate_video(path, frames, width, height, fps):
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(actual_fps) or abs(actual_fps - fps) > .01:
            raise ValueError(f"Unreadable video or unexpected FPS: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(f"Unexpected video dimensions: {path}")
            count += 1
    finally:
        capture.release()
    if count != frames:
        raise ValueError(f"Video has {count} frames, expected {frames}: {path}")
    return count


def cached_video(path, input_fingerprint, encode, *, frames, width, height, fps, force=False):
    """Reuse only matching inputs and unchanged, previously validated bytes."""
    path = Path(path)
    metadata_path = path.with_suffix(".cache.json")
    expected = {"schema": 1, "input_fingerprint": input_fingerprint,
                "frames": frames, "width": width, "height": height, "fps": fps}
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (all(previous.get(k) == v for k, v in expected.items())
                    and previous.get("all_frames_decoded") is True
                    and file_sha256(path) == previous.get("video_sha256"))
    except (OSError, ValueError, TypeError, AttributeError):
        reusable = False
    if reusable and not force:
        return {**previous, "cache_hit": True}
    with atomic_path(path) as staged:
        encode(staged)
        validate_video(staged, frames, width, height, fps)
        record = {**expected, "video_sha256": file_sha256(staged),
                  "all_frames_decoded": True}
    atomic_json(metadata_path, record)
    return {**record, "cache_hit": False}
