import shutil
import subprocess

import pytest

from pingpong_event_pipeline.artifacts import cached_video, file_sha256


@pytest.fixture
def encode():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for video artifact integration checks")

    def make(path, color="blue", frames=3):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                        "-f", "lavfi", "-i", f"color=c={color}:s=32x32:r=25",
                        "-frames:v", str(frames), "-c:v", "libx264", "-threads", "1",
                        "-pix_fmt", "yuv420p", str(path)], check=True)
    return make


def render(path, key, encode, **options):
    return cached_video(path, key, encode, frames=3, width=32, height=32, fps=25, **options)


def test_matching_input_reuses_only_unchanged_validated_video(tmp_path, encode):
    path = tmp_path / "review.mp4"
    assert not render(path, "inputs-a", encode)["cache_hit"]

    def unexpected(_):
        pytest.fail("unchanged validated inputs should not be re-encoded")
    assert render(path, "inputs-a", unexpected)["cache_hit"]
    before = file_sha256(path)
    changed = render(path, "inputs-b", lambda p: encode(p, color="red"))
    assert not changed["cache_hit"] and file_sha256(path) != before
    assert changed["input_fingerprint"] == "inputs-b"


def test_corruption_or_missing_provenance_triggers_rebuild(tmp_path, encode):
    path = tmp_path / "review.mp4"
    render(path, "a", encode)
    path.write_bytes(b"interrupted or externally replaced video")
    assert not render(path, "a", encode)["cache_hit"]
    path.with_suffix(".cache.json").unlink()
    assert not render(path, "a", encode)["cache_hit"]


@pytest.mark.parametrize("failure", ["producer", "wrong_frame_count"])
@pytest.mark.parametrize("force", [False, True])
def test_failed_replacement_preserves_previous_video_and_cache(tmp_path, encode, failure, force):
    path = tmp_path / "review.mp4"
    render(path, "a", encode)
    before = path.read_bytes()
    metadata = path.with_suffix(".cache.json").read_bytes()

    def fail(staged):
        if failure == "wrong_frame_count":
            encode(staged, frames=2)
        else:
            staged.write_bytes(b"unfinished")
            raise RuntimeError("encoder failed")

    with pytest.raises((RuntimeError, ValueError)):
        render(path, "b", fail, force=force)
    assert path.read_bytes() == before
    assert path.with_suffix(".cache.json").read_bytes() == metadata
    assert not list(tmp_path.glob(".review-*"))
