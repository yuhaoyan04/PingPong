import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from export_supported_ball_crops import supported_runs, candidate_runs, window_support


def test_does_not_bridge_unsupported_or_missing_timestamps():
    rows = np.zeros((10, 8))
    rows[:, 0] = np.array([0, 1000, 2000, 3000, 4000, 6000, 7000, 8000, 9000, 10000])
    rows[:, 5] = 1
    rows[3, 5] = 0
    runs = supported_runs(rows)
    assert [list(r[:, 0]) for r in runs] == [[0, 1000, 2000], [6000, 7000, 8000, 9000, 10000]]
    candidates = candidate_runs(rows)
    assert [list(r[:, 0]) for r in candidates] == [[0,1000,2000,3000,4000], [6000,7000,8000,9000,10000]]
    assert window_support(candidates[0], 2000)
    assert not window_support(candidates[0], 2500)
    assert not window_support(candidates[0], 3000)
