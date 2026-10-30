import numpy as np

from pingpong_event_pipeline.crescent import _Lobe, _robust_lobe_circle


def test_robust_lobe_circle_ignores_sparse_motion_trail():
    angles_off = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    angles_on = np.linspace(-np.pi / 2, np.pi / 2, 200)
    center = np.array([120.0, 80.0])
    radius = 24.0
    off_points = center + radius * np.column_stack((np.cos(angles_off), np.sin(angles_off)))
    on_points = center + radius * np.column_stack((np.cos(angles_on), np.sin(angles_on)))
    # A few old events trail far behind the ball.
    off_points = np.vstack((off_points, np.column_stack((np.linspace(20, 70, 6),
                                                         np.full(6, 80.0)))))
    off = _Lobe(0, np.array([103.0, 80.0]), off_points, len(off_points), 1.0)
    on = _Lobe(1, np.array([137.0, 80.0]), on_points, len(on_points), 1.0)
    x, y, fitted_radius = _robust_lobe_circle(off, on)
    assert np.linalg.norm(np.array([x, y]) - center) < 2.0
    assert abs(fitted_radius - radius) < 2.0
