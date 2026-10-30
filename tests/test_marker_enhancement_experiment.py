import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from marker_enhancement_experiment import color_counts, surfaces


def test_normalized_surfaces_warp_and_time_window():
    dtype = [('x', 'u2'), ('y', 'u2'), ('polarity', 'u1'), ('timestamp', 'u8')]
    events = np.array([(100, 100, 0, 1000), (105, 100, 1, 2000), (110, 100, 1, 3000)], dtype=dtype)
    ts = np.array([1000, 3000], dtype=np.int64)
    centers = np.array([[100., 100.], [110., 100.]])
    radii = np.array([10., 10.])
    surface, counts = surfaces(events, 3000, 2000, ts, centers, radii)
    assert counts == {'inside_ball': 2, 'inside_078': 2}
    assert surface[0, 96, 96] == 1 and surface[1, 96, 96] == 1
    for mode in ('RAW', 'DENSITY', 'LOCAL'):
        assert color_counts(surface, mode).shape == (288, 288, 3)
