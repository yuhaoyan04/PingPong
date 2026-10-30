import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_raw_group import prepare


def test_only_actual_detections_seed_generic_raw_refinement(tmp_path):
    raw, video, path = tmp_path / "input.raw", tmp_path / "video.mp4", tmp_path / "path.npz"
    raw.touch(); video.touch()
    timestamps = 1_000_000 + np.arange(6) * 5000
    np.savez(path, detection_timestamps_us=timestamps,
             measured_circles_px=np.array([[50 + i, 60, 20] for i in range(6)]),
             measured_scores=[1, 1, 0, 1, 1, 0])
    output = tmp_path / "prepared"
    prepare(path, raw, video, "F3E", "serve10_F3E", output, 1_000_000)
    with np.load(output / "coarse/serve10_F3E/candidate_tracks.npz") as data:
        np.testing.assert_array_equal(data["new"][:, 0], timestamps[[0, 1, 3, 4]])
    source = json.loads((output / "groups/serve10_F3E/source.json").read_text())
    assert source["frames"] == 4
    assert source["video_step_us"] == 5000
    with pytest.raises(FileExistsError):
        prepare(path, raw, video, "F3E", "serve10_F3E", output, 1_000_000)


def test_refuse_prediction_only_seeds(tmp_path):
    raw, video, path = tmp_path / "input.raw", tmp_path / "video.mp4", tmp_path / "path.npz"
    raw.touch(); video.touch()
    np.savez(path, detection_timestamps_us=np.arange(6) * 5000,
             measured_circles_px=np.ones((6, 3)), measured_scores=np.zeros(6))
    with pytest.raises(ValueError, match="actual ball detections"):
        prepare(path, raw, video, "F59", "serve10_F59", tmp_path / "output", 0)
