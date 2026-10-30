import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "preflight_spin_capture",
    Path(__file__).parents[1] / "tools" / "preflight_spin_capture.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_longest_valid_segment_respects_gaps_duration_and_selected_rows():
    timestamps = np.array([0, 1000, 2000, 9000, 10_000, 11_000, 12_000, 13_000])
    selected = np.array([0, 0, -1, 1, 1, 1, 1, 1])
    assert module.longest_valid_segment(timestamps, selected, 2500, 3000) == (
        4000, 5, 3, 7)


def test_longest_valid_segment_rejects_short_or_missing_data():
    timestamps = np.array([0, 1000, 2000])
    assert module.longest_valid_segment(timestamps, np.array([-1, -1, -1]), 2000, 1000) is None
    assert module.longest_valid_segment(timestamps, np.array([0, 0, 0]), 2000, 3000) is None
