import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "measure_trial", Path(__file__).parents[1] / "tools" / "measure_trial.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_split_motion_segments_ignores_short_groups_and_preserves_long_flights():
    timestamps = np.arange(20, dtype=np.int64) * 1000
    selected = np.full(20, -1, np.int32)
    selected[[0, 1, 2, 10, 11, 13, 14, 15, 16, 17]] = 0
    assert module.split_motion_segments(timestamps, selected, 4000, 5000) == [
        (10, 17, 7000, 7)]
