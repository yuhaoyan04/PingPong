import numpy as np

from pingpong_event_pipeline.events import EVENT_DTYPE, trajectory_compensated_ball_patch


def test_trajectory_compensation_preserves_ball_relative_location():
    path_t = np.array([0, 1000, 2000], np.int64)
    centers = np.array([[100, 50], [110, 50], [120, 50]], np.float64)
    radii = np.array([20, 25, 30], np.float64)
    events = np.empty(3, dtype=EVENT_DTYPE)
    events["timestamp"] = path_t
    events["x"] = [110, 122, 135]  # approximately centre + 0.5 * radius
    events["y"] = 50
    events["polarity"] = 1
    patch = trajectory_compensated_ball_patch(
        events, path_t, centers, radii, 100, 2000, 10000, 3000)
    active_y, active_x = np.where(patch[1] > 0)
    assert len(active_x) <= 2
    assert np.all(np.abs(active_x - 75) <= 1)
    assert np.all(active_y == 50)
