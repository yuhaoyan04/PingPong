import numpy as np
import pytest

from pingpong_event_pipeline.ball_event_refine import robust_seed, warped_local_image, refine_track, refine_track_adaptive
from pingpong_event_pipeline.events import EVENT_DTYPE


def test_warp_moves_old_events_forward_along_velocity():
    events = np.array([(10, 20, 1, 900), (11, 20, 0, 1000)], dtype=EVENT_DTYPE)
    image = warped_local_image(events, 1000, (11, 20), (.01, 0), 10)
    assert image[10, 10] > 0
    assert np.count_nonzero(image) == 1


def test_no_events_never_reported_as_supported():
    points = np.array([[i*5000, 100+i*10, 200+i*5, 35] for i in range(6)], float)
    out, keep = refine_track(np.empty(0, dtype=EVENT_DTYPE), points, step_us=5000)
    assert keep.sum() >= 6 and len(out) == 6
    assert not np.any(out[:, 5])


def test_unordered_and_short_seed_rejected():
    with pytest.raises(ValueError):
        robust_seed(np.zeros((3, 4)))


def test_adaptive_radius_can_move_beyond_old_five_pixel_gate():
    points = np.array([[i*5000, 200, 200, 40] for i in range(6)], float)
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
    ring = [(int(round(200+54*np.cos(a))), int(round(200+54*np.sin(a))), 1, int(t-100))
            for t in range(5000, 26000, 5000) for a in angles]
    events = np.array(ring, dtype=EVENT_DTYPE)
    out, _ = refine_track_adaptive(events, points, step_us=5000)
    assert np.any((out[:, 5] > 0) & (out[:, 3] > 45))


def test_adaptive_empty_events_are_not_supported():
    points = np.array([[i*5000, 100+i, 200, 35] for i in range(6)], float)
    out, _ = refine_track_adaptive(np.empty(0, dtype=EVENT_DTYPE), points, step_us=5000)
    assert not out[:, 5].any()
