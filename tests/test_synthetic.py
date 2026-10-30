import tempfile
from pathlib import Path

import cv2
import numpy as np
import pingpong_event_pipeline.events as events_module

from pingpong_event_pipeline.detector import ClassicalBallDetector, HoughCircleBallDetector
from pingpong_event_pipeline.models import Detection2D
from pingpong_event_pipeline.events import (EVENT_DTYPE, IncrementalTimeSurface,
                                             motion_compensated_ball_patch,
                                             motion_compensated_sae, sae)
from pingpong_event_pipeline.stereo import StereoCalibration, select_stereo_detection_pair
from pingpong_event_pipeline.spin import (ImagePlaneSpinEstimator, fuse_optical_axis_spin,
                                           solve_sphere_angular_velocity,
                                           solve_perspective_sphere_angular_velocity,
                                           solve_rigid_sphere_rotation,
                                           solve_constant_sphere_rotation,
                                           _rotate_vectors)
from pingpong_event_pipeline.trajectory import TrajectoryFilter3D
from pingpong_event_pipeline.tracking import BallTracker2D
from pingpong_event_pipeline.quality import marker_structure_metrics, summarize_spin_segments
from tools.validate_spin_roll import robust_segment_velocity
from pingpong_event_pipeline.offline_path import (fit_smooth_candidate_path,
                                                   fit_smooth_stereo_candidate_path)
from pingpong_event_pipeline.stereo_spin import (match_sphere_marker_sets,
                                                  triangulate_sphere_markers,
                                                  estimate_unordered_sphere_rotation,
                                                  estimate_sphere_template_pose,
                                                  SphereTemplateMatcher,
                                                  SphereTemplatePoseTracker,
                                                  estimate_spin_from_template_poses,
                                                  build_labeled_sphere_template,
                                                  audit_sphere_template_symmetry,
                                                  CalibratedTemplateSpinStream)


def test_sae_and_detector():
    events = np.zeros(240, dtype=EVENT_DTYPE)
    angles = np.linspace(0, 2 * np.pi, len(events), endpoint=False)
    events["x"] = np.rint(80 + 12 * np.cos(angles)).astype(np.uint16)
    events["y"] = np.rint(60 + 12 * np.sin(angles)).astype(np.uint16)
    events["polarity"] = np.arange(len(events)) % 2
    events["timestamp"] = 9990
    surface = sae(events, 160, 120, 10000, 2500)
    cfg = dict(threshold_percentile=90, minimum_threshold=10, min_radius_px=3,
               max_radius_px=30, min_area_px=10, max_area_px=2000,
               max_aspect_ratio=3, roi_half_size_px=50)
    detected = ClassicalBallDetector(cfg).detect(surface, 10000)
    assert detected and abs(detected[0].x - 80) < 2 and abs(detected[0].y - 60) < 2


def test_hough_detector_finds_sparse_circle():
    image = np.zeros((2, 180, 240), np.float32)
    cv2.circle(image[1], (130, 85), 28, 1.0, 2)
    cfg = dict(min_radius_px=12, max_radius_px=60, roi_half_size_px=80,
               hough_scale=1.0, hough_vote_threshold=15, minimum_ring_score=0.05)
    detected = HoughCircleBallDetector(cfg).detect(image, 123)
    assert detected
    best = min(detected, key=lambda d: np.hypot(d.x - 130, d.y - 85))
    assert np.hypot(best.x - 130, best.y - 85) < 4
    assert abs(best.radius - 28) < 4


def test_hough_detector_uses_fast_predictive_ring_update():
    image = np.zeros((2, 180, 240), np.float32)
    cv2.circle(image[1], (134, 82), 29, 1.0, 2)
    cfg = dict(min_radius_px=12, max_radius_px=60, roi_half_size_px=80,
               hough_scale=1.0, hough_vote_threshold=15, minimum_ring_score=0.05,
               predictive_refinement=True, predictive_min_ring_score=0.03)
    detected = HoughCircleBallDetector(cfg).detect(image, 123, (132, 84, 28))
    assert len(detected) == 1
    assert np.hypot(detected[0].x - 134, detected[0].y - 82) < 2


def test_predictive_ring_search_recovers_from_motion_offset():
    image = np.zeros((2, 180, 240), np.float32)
    cv2.circle(image[1], (146, 76), 29, 1.0, 2)
    cfg = dict(min_radius_px=12, max_radius_px=60, roi_half_size_px=70,
               hough_scale=1.0, hough_vote_threshold=15, minimum_ring_score=0.05,
               predictive_refinement=True, predictive_min_ring_score=0.03,
               predictive_search_px=16, predictive_search_step_px=2,
               predictive_radius_search_px=2, predictive_radius_step_px=2)
    detected = HoughCircleBallDetector(cfg).detect(image, 123, (134, 84, 29))
    assert len(detected) == 1
    assert np.hypot(detected[0].x - 146, detected[0].y - 76) < 3


def test_full_resolution_circle_refinement_is_subpixel_accurate():
    image = np.zeros((180, 240), np.uint8)
    expected = np.array([123.4, 81.7]); radius = 31.2
    angles = np.linspace(0.15, 5.8, 500)
    x = np.rint(expected[0] + radius * np.cos(angles)).astype(int)
    y = np.rint(expected[1] + radius * np.sin(angles)).astype(int)
    image[y, x] = 255
    cx, cy, fitted_radius = HoughCircleBallDetector._refine_circle(image, 121.0, 83.0, 30.0)
    assert np.linalg.norm(np.array([cx, cy]) - expected) < 0.7
    assert abs(fitted_radius - radius) < 0.7


def test_internal_dot_constellation_detector_excludes_ball_silhouette():
    patch = np.zeros((128, 128), np.uint8)
    cv2.circle(patch, (64, 64), 61, 255, 2)
    for center, radius in [((45, 48), 6), ((75, 43), 4), ((70, 76), 8), ((43, 77), 3)]:
        cv2.circle(patch, center, radius, 255, 2)
    cfg = dict(max_rpm=12000, minimum_quality=0.0, dot_vote_threshold=5)
    dots = ImagePlaneSpinEstimator(cfg)._dot_centers(patch)
    assert len(dots) >= 4
    assert np.all(np.hypot(dots[:, 0] - 64, dots[:, 1] - 64) < 52)


def test_blob_marker_detector_finds_compact_internal_event_regions():
    patch = np.zeros((128, 128), np.uint8)
    for center in ((43, 48), (75, 43), (70, 76), (43, 77)):
        cv2.ellipse(patch, center, (5, 3), 25, 20, 300, 255, 2)
    cv2.circle(patch, (64, 64), 61, 255, 2)
    estimator = ImagePlaneSpinEstimator({
        "dot_detector": "blobs", "maximum_dots": 8,
        "dot_blob_threshold": 20, "dot_mask_radius_fraction": .72,
        "dot_blob_min_area_px": 4, "dot_blob_max_area_px": 200})
    dots = estimator._dot_centers(patch)
    assert len(dots) == 4
    for expected in ((43, 48), (75, 43), (70, 76), (43, 77)):
        assert np.min(np.linalg.norm(dots[:, :2] - expected, axis=1)) < 3


def test_motion_compensated_sae_collapses_linear_trail():
    events = np.zeros(11, dtype=EVENT_DTYPE)
    events["timestamp"] = np.arange(0, 1100, 100)
    events["x"] = 40 + np.arange(11)  # 10,000 px/s for 1 ms
    events["y"] = 30
    events["polarity"] = 1
    uncompensated = sae(events, 100, 60, 1000, 10_000)
    compensated = motion_compensated_sae(
        events, 100, 60, 1000, 10_000, (10_000.0, 0.0), 1000)
    assert np.count_nonzero(uncompensated[1]) == 11
    assert np.count_nonzero(compensated[1]) == 1
    assert compensated[1, 30, 50] > 0.99


def test_motion_compensated_ball_patch_collapses_linear_trail():
    events = np.zeros(11, dtype=EVENT_DTYPE)
    events["timestamp"] = np.arange(0, 1100, 100)
    events["x"] = 40 + np.arange(11); events["y"] = 30; events["polarity"] = 1
    patch = motion_compensated_ball_patch(
        events, (50, 30), 20, 80, 1000, 10_000, (10_000, 0), 1000)
    assert np.count_nonzero(patch[1]) == 1
    assert patch[1, 40, 40] > 0.99


def test_motion_compensated_ball_patch_excludes_square_corner_background():
    events = np.zeros(2, dtype=EVENT_DTYPE)
    events["timestamp"] = 1000
    events["polarity"] = 1
    events[0]["x"], events[0]["y"] = 50, 30  # ball centre
    events[1]["x"], events[1]["y"] = 68, 48  # inside square, outside r=20 disc
    patch = motion_compensated_ball_patch(
        events, (50, 30), 20, 80, 1000, 10_000, (0, 0), 1000)
    assert np.count_nonzero(patch[1]) == 1
    assert patch[1, 40, 40] > 0.99


def test_motion_compensated_ball_patch_numpy_fallback_excludes_background(monkeypatch):
    monkeypatch.setattr(events_module, "NUMBA_AVAILABLE", False)
    events = np.zeros(2, dtype=EVENT_DTYPE)
    events["timestamp"] = 1000
    events["polarity"] = 1
    events[0]["x"], events[0]["y"] = 50, 30
    events[1]["x"], events[1]["y"] = 68, 48
    patch = motion_compensated_ball_patch(
        events, (50, 30), 20, 80, 1000, 10_000, (0, 0), 1000)
    assert np.count_nonzero(patch[1]) == 1
    assert patch[1, 40, 40] > 0.99


def test_marker_structure_metrics_follow_moving_ball_and_exclude_background():
    events = np.zeros(42, dtype=EVENT_DTYPE)
    events["timestamp"] = 1000
    # Balanced, concentrated marker activity around the tracked centre.
    events["x"][:40] = np.tile(np.arange(48, 53), 8)
    events["y"][:40] = np.repeat(np.arange(28, 36), 5)
    events["polarity"][:40] = np.arange(40) % 2
    # Square-corner background is outside the inner ball disc.
    events["x"][40:] = [68, 69]; events["y"][40:] = [48, 49]
    metrics = marker_structure_metrics(events, 1000, (50, 30), (0, 0), 20)
    assert metrics["events"] == 40
    assert metrics["on"] == metrics["off"] == 20
    assert metrics["polarity_balance"] == 1.0
    assert metrics["density_peak_to_mean"] > 2.0


def test_incremental_time_surface_matches_full_sae():
    events = np.zeros(100, dtype=EVENT_DTYPE)
    events["x"] = np.arange(100) % 20
    events["y"] = np.arange(100) % 15
    events["polarity"] = np.arange(100) % 2
    events["timestamp"] = np.arange(100) * 10 + 1000
    accumulator = IncrementalTimeSurface(32, 24)
    accumulator.ingest(events[:43]); accumulator.ingest(events[43:])
    expected = sae(events, 32, 24, 2000, 500)
    actual = accumulator.surface(2000, 500, 2000)
    assert np.allclose(actual, expected)


def test_stereo_triangulation():
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    R = np.eye(3); T = np.array([[-0.2], [0.0], [0.0]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stereo.npz"
        np.savez(p, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=R, T=T,
                 P1=P1, P2=P2, F=np.array([[0,0,0],[0,0,1],[0,-1,0.]]))
        calibration = StereoCalibration(p)
        point = np.array([0.1, -0.05, 2.0])
        def project(P):
            q = P @ np.r_[point, 1.0]; return tuple(q[:2] / q[2])
        estimate, error = calibration.triangulate(project(P1), project(P2))
        assert np.allclose(estimate, point, atol=1e-5)
        assert error < 1e-5


def test_stereo_triangulation_converts_rectified_coordinates_back():
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    angle = np.deg2rad(8.0)
    R1 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                   [np.sin(angle), np.cos(angle), 0.0],
                   [0.0, 0.0, 1.0]])
    R2 = np.eye(3)
    R = R1.copy()
    T = np.array([[-0.2], [0.0], [0.0]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((np.eye(3), T))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stereo.npz"
        np.savez(p, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=R, T=T,
                 R1=R1, R2=R2, P1=P1, P2=P2,
                 F=np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0.]]))
        calibration = StereoCalibration(p)
        point = np.array([0.1, -0.05, 2.0])

        def camera_project(X):
            q = K @ X
            return q[:2] / q[2]

        left_xy = camera_project(point)
        right_xy = camera_project(R @ point + T[:, 0])
        estimate, error = calibration.triangulate(left_xy, right_xy)
        assert np.allclose(estimate, point, atol=1e-5)
        assert error < 1e-5


def test_spin_vector_fusion_coordinate_transform():
    left = np.array([10.0, 20.0, 30.0])
    # 右相机绕 y 轴旋转 90 度；传入的是 right -> left。
    rtl = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    right = rtl.T @ left
    fused, quality = fuse_optical_axis_spin(left, right, rtl, 0.9, 0.9)
    assert np.allclose(fused, left)
    assert quality > 0.89


def test_sphere_angular_velocity_solver_recovers_known_vector():
    rng = np.random.default_rng(7)
    xy0 = rng.uniform(-0.65, 0.65, size=(40, 2))
    xy0 = xy0[np.sum(xy0 * xy0, axis=1) < 0.75 ** 2]
    z = -np.sqrt(1.0 - np.sum(xy0 * xy0, axis=1))
    points = np.column_stack((xy0, z))
    expected = np.array([30.0, -20.0, 50.0])
    dt = 0.0005
    velocity = np.cross(np.broadcast_to(expected, points.shape), points)
    xy1 = xy0 + velocity[:, :2] * dt

    estimated, inliers, residual, condition = solve_sphere_angular_velocity(xy0, xy1, dt)

    assert np.count_nonzero(inliers) == len(xy0)
    assert np.allclose(estimated, expected, atol=1e-8)
    assert residual < 1e-9
    assert condition < 10


def test_sphere_angular_velocity_solver_rejects_flow_outliers():
    rng = np.random.default_rng(19)
    xy0 = rng.uniform(-0.7, 0.7, size=(50, 2))
    xy0 = xy0[np.sum(xy0 * xy0, axis=1) < 0.8 ** 2]
    z = -np.sqrt(1.0 - np.sum(xy0 * xy0, axis=1))
    points = np.column_stack((xy0, z))
    expected = np.array([-25.0, 40.0, 70.0])
    dt = 0.001
    xy1 = xy0 + np.cross(np.broadcast_to(expected, points.shape), points)[:, :2] * dt
    outliers = np.arange(0, len(xy0), 5)
    xy1[outliers] += rng.uniform(-0.2, 0.2, size=(len(outliers), 2))
    estimated, inliers, residual, _ = solve_sphere_angular_velocity(xy0, xy1, dt)
    assert np.count_nonzero(~inliers) >= len(outliers) - 1
    assert np.allclose(estimated, expected, atol=1e-6)
    assert residual < 1e-6


def test_finite_sphere_rotation_recovers_fast_spin_with_outliers():
    rng = np.random.default_rng(2026)
    points = rng.normal(size=(80, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points[points[:, 2] < -0.1][:40]
    axis = np.array([0.35, -0.2, 0.915])
    axis /= np.linalg.norm(axis)
    expected = axis * 420.0
    dt = 0.0025  # about 60 degrees per update
    rotation, _ = cv2.Rodrigues(expected * dt)
    moved = (rotation @ points.T).T
    moved += rng.normal(0, 0.002, moved.shape)
    moved /= np.linalg.norm(moved, axis=1, keepdims=True)
    outliers = np.arange(0, len(points), 7)
    moved[outliers] = rng.normal(size=(len(outliers), 3))
    moved[outliers] /= np.linalg.norm(moved[outliers], axis=1, keepdims=True)

    estimated, inliers, residual, condition = solve_rigid_sphere_rotation(
        points, moved, dt, angular_threshold_rad=0.02)

    assert np.count_nonzero(~inliers) >= len(outliers) - 1
    assert np.allclose(estimated, expected, rtol=0.01, atol=1.0)
    assert residual < 2.0
    assert condition < 10.0


def test_stereo_perspective_spin_coordinates_agree():
    rng = np.random.default_rng(23)
    angle = np.deg2rad(7.0)
    master_to_slave = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angle), 0.0, np.cos(angle)]])
    omega_master = np.array([35.0, -55.0, 80.0])
    ball_radius, dt = 0.02, 0.0001

    def camera_measurement(center, omega, fx, fy):
        directions = rng.normal(size=(200, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions[directions[:, 2] < -0.15][:60]
        points0 = center + ball_radius * directions
        moved = directions + np.cross(np.broadcast_to(omega, directions.shape), directions) * dt
        moved /= np.linalg.norm(moved, axis=1, keepdims=True)
        points1 = center + ball_radius * moved
        projection0 = np.column_stack((fx * points0[:, 0] / points0[:, 2],
                                       fy * points0[:, 1] / points0[:, 2]))
        projection1 = np.column_stack((fx * points1[:, 0] / points1[:, 2],
                                       fy * points1[:, 1] / points1[:, 2]))
        center_px = np.array([fx * center[0] / center[2], fy * center[1] / center[2]])
        radius_px = 0.5 * (fx + fy) * ball_radius / center[2]
        return projection0, projection1, center_px, radius_px

    master_xy = camera_measurement(np.array([0.02, -0.01, 1.1]), omega_master, 2690.0, 2689.0)
    omega_slave = master_to_slave @ omega_master
    slave_xy = camera_measurement(np.array([-0.05, -0.06, 1.05]), omega_slave, 2578.0, 2582.0)
    K1 = np.array([[2690., 0, 0], [0, 2689., 0], [0, 0, 1.]])
    K2 = np.array([[2578., 0, 0], [0, 2582., 0], [0, 0, 1.]])
    estimated_master = solve_perspective_sphere_angular_velocity(
        master_xy[0], master_xy[1], master_xy[2], master_xy[2], master_xy[3], master_xy[3], K1, dt)[0]
    estimated_slave = solve_perspective_sphere_angular_velocity(
        slave_xy[0], slave_xy[1], slave_xy[2], slave_xy[2], slave_xy[3], slave_xy[3], K2, dt)[0]
    slave_in_master = master_to_slave.T @ estimated_slave
    assert np.allclose(estimated_master, omega_master, rtol=0.02, atol=1.0)
    assert np.allclose(slave_in_master, omega_master, rtol=0.02, atol=1.0)


def test_stereo_pair_selection_rejects_high_score_epipolar_distractor():
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    R = np.eye(3); T = np.array([[-0.2], [0.0], [0.0]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stereo.npz"
        np.savez(p, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=R, T=T,
                 P1=P1, P2=P2, F=np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0.]]))
        calibration = StereoCalibration(p)
        left = [Detection2D(1, 400, 250, 20, 0.8, 100)]
        right = [Detection2D(1, 360, 330, 20, 0.99, 200),
                 Detection2D(1, 340, 250, 19, 0.75, 90)]
        pair = select_stereo_detection_pair(left, right, calibration, 4.0)
        assert pair is not None
        assert pair[1].y == 250


def test_camera_frame_trajectory_does_not_invent_gravity_direction():
    trajectory = TrajectoryFilter3D(dict(
        gravity_vector_camera_mps2=[0.0, 0.0, 0.0], gravity_mps2=9.80665,
        drag_coefficient=0.0, process_noise=0.01, measurement_noise_m=0.001,
    ))
    point = np.array([0.1, -0.2, 1.0])
    trajectory.update(1_000_000, point)
    for timestamp in range(1_010_000, 2_000_000, 10_000):
        position, velocity = trajectory.update(timestamp, point)
    assert np.linalg.norm(position - point) < 1e-6
    assert np.linalg.norm(velocity) < 1e-6


def test_spin_segment_summary_rejects_outliers_and_recovers_axis():
    rng = np.random.default_rng(11)
    expected = np.array([15.0, -30.0, 45.0])
    vectors = expected + rng.normal(0, 0.7, size=(60, 3))
    vectors[[5, 17, 44]] = rng.normal(0, 80, size=(3, 3))
    timestamps = np.arange(60) * 2500 + 1_000_000
    summaries = summarize_spin_segments(timestamps, vectors, np.full(60, 0.8),
                                        np.full(60, 0.1), minimum_samples=20)
    assert len(summaries) == 1
    result = summaries[0]
    assert result["internally_consistent"]
    assert result["samples"] >= 55
    expected_axis = expected / np.linalg.norm(expected)
    assert np.dot(result["spin_axis_camera"], expected_axis) > 0.999


def test_robust_segment_velocity_rejects_adjacent_depth_jitter():
    rng = np.random.default_rng(91)
    t = np.arange(240) * 2500
    expected = np.array([0.6, -0.15, 0.05])
    positions = np.array([0.1, 0.2, 1.0]) + t[:, None] * 1e-6 * expected
    positions += rng.normal(0, [0.001, 0.001, 0.004], positions.shape)
    positions[::31] += rng.normal(0, 0.04, (len(positions[::31]), 3))
    velocity, rms, duration = robust_segment_velocity(t, positions)
    assert np.allclose(velocity, expected, atol=0.025)
    assert rms < 0.01
    assert duration > 0.5


def test_prediction_only_frames_do_not_count_as_missed_detections():
    tracker = BallTracker2D(max_missed=2)
    tracker.update(1000, [Detection2D(1000, 50, 40, 12, 0.9, 100)])
    for timestamp in range(1500, 10_000, 500):
        state = tracker.advance(timestamp)
        assert state is not None
    assert tracker.missed == 0
    assert tracker.initialized


def test_offline_candidate_path_rejects_high_score_arc_distractors():
    rng = np.random.default_rng(88)
    timestamps = np.arange(120, dtype=np.int64) * 1000 + 2_000_000
    u = np.arange(120) / 119.0
    expected = np.column_stack((1250 - 1160 * u + 20 * u * u,
                                30 + 350 * u + 25 * u * u))
    candidates = []
    for i, point in enumerate(expected):
        values = [Detection2D(int(timestamps[i]), *(point + rng.normal(0, 2, 2)),
                              70 + 8 * u[i], 0.45, 200)]
        for _ in range(9):
            values.append(Detection2D(int(timestamps[i]), *rng.uniform([0, 0], [1280, 720]),
                                      rng.uniform(15, 45), 0.8, 80))
        candidates.append(values)
    path = fit_smooth_candidate_path(timestamps, candidates, hypotheses=4000)
    error = np.linalg.norm(path.centers_px - expected, axis=1)
    assert np.median(error) < 3.0
    assert np.percentile(error, 95) < 6.0


def test_joint_stereo_path_rejects_independent_image_distractors():
    rng = np.random.default_rng(889)
    K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
    R = np.eye(3); T = np.array([[-0.1], [0.0], [0.0]])
    P1 = K @ np.hstack((R, np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))
    tx = np.array([[0., 0., 0.], [0., 0., 0.1], [0., -0.1, 0.]])
    F = np.linalg.inv(K).T @ tx @ np.linalg.inv(K)
    timestamps = np.arange(80, dtype=np.int64) * 1000 + 3_000_000
    time_s = np.arange(80) * 0.001
    expected = np.column_stack((-0.12 + 3.0 * time_s,
                                0.02 + 0.4 * time_s - 2.0 * time_s ** 2,
                                1.2 + 0.3 * time_s))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "stereo.npz"
        np.savez(path, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=R, T=T,
                 P1=P1, P2=P2, F=F)
        calibration = StereoCalibration(path)
        left_bins = []; right_bins = []
        for timestamp, point in zip(timestamps, expected):
            left_xy = (K @ point)[:2] / (K @ point)[2]
            right_point = R @ point + T[:, 0]
            right_xy = (K @ right_point)[:2] / (K @ right_point)[2]
            left_radius = K[0, 0] * 0.020 / point[2]
            right_radius = K[0, 0] * 0.020 / right_point[2]
            left = [Detection2D(int(timestamp), *(left_xy + rng.normal(0, .5, 2)),
                                left_radius, .45, 100)]
            right = [Detection2D(int(timestamp), *(right_xy + rng.normal(0, .5, 2)),
                                 right_radius, .45, 100)]
            for _ in range(7):
                left.append(Detection2D(int(timestamp), *rng.uniform([0, 0], [640, 480]),
                                        rng.uniform(7, 28), .85, 50))
                right.append(Detection2D(int(timestamp), *rng.uniform([0, 0], [640, 480]),
                                         rng.uniform(7, 28), .85, 50))
            left_bins.append(left); right_bins.append(right)
        result = fit_smooth_stereo_candidate_path(
            timestamps, left_bins, right_bins, calibration,
            hypotheses=3000, inlier_threshold_m=.025)
        error = np.linalg.norm(result.points_m - expected, axis=1)
        assert np.median(error) < .005
        assert np.percentile(error, 95) < .012
        assert np.count_nonzero(result.selected >= 0) > 70


def test_stereo_sphere_constraint_rejects_false_marker_pairs():
    K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
    R = np.eye(3); T = np.array([[-0.1], [0.0], [0.0]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "stereo.npz"
        np.savez(path, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=R, T=T,
                 P1=P1, P2=P2, F=np.eye(3))
        calibration = StereoCalibration(path)
        center = np.array([0.02, -0.01, 1.0]); radius = 0.02
        directions = np.array([[0.3, 0.2, -0.932], [-0.4, 0.1, -0.911], [0.1, -0.5, -0.86]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = center + radius * directions
        project = lambda X: (K @ X)[:2] / (K @ X)[2]
        left = np.array([project(p) for p in points])
        right = np.array([project(R @ p + T[:, 0]) for p in points])
        left = np.vstack((left, [[100., 100.], [500., 350.]]))
        right = np.vstack((right, [[600., 120.], [50., 420.]]))
        result = triangulate_sphere_markers(left, right, calibration, center, radius)
        assert len(result.directions) == 3
        assert np.max(result.reprojection_error_px) < 1e-6
        matched0, matched1 = match_sphere_marker_sets(directions, result.directions, 0.05)
        assert len(matched0) == 3


def test_unordered_sphere_rotation_recovers_pose_with_outliers():
    rng = np.random.default_rng(444)
    source = rng.normal(size=(9, 3)); source /= np.linalg.norm(source, axis=1, keepdims=True)
    expected_vector = np.array([0.04, -0.06, 0.03])
    expected_rotation, _ = cv2.Rodrigues(expected_vector)
    target = (expected_rotation @ source.T).T + rng.normal(0, 0.002, source.shape)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    target = target[rng.permutation(len(target))]
    source = np.vstack((source, rng.normal(size=(3, 3))))
    source[-3:] /= np.linalg.norm(source[-3:], axis=1, keepdims=True)
    target = np.vstack((target, rng.normal(size=(4, 3))))
    target[-4:] /= np.linalg.norm(target[-4:], axis=1, keepdims=True)
    rotation, old, new, residual = estimate_unordered_sphere_rotation(
        source, target, maximum_rotation_rad=0.15, inlier_chord_distance=0.02,
        hypotheses=3000)
    recovered, _ = cv2.Rodrigues(rotation)
    assert len(old) >= 8
    assert np.allclose(recovered.reshape(3), expected_vector, atol=0.01)
    assert residual < 0.01


def test_absolute_sphere_template_pose_recovers_large_rotation_and_indices():
    rng = np.random.default_rng(446)
    template = rng.normal(size=(10, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    expected_vector = np.array([.7, -.4, .5])
    expected_rotation, _ = cv2.Rodrigues(expected_vector)
    retained = np.array([0, 1, 3, 4, 6, 7, 9])
    observed = (expected_rotation @ template[retained].T).T
    observed += rng.normal(0, .002, observed.shape)
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    observed = np.vstack((observed, rng.normal(size=(3, 3))))
    observed[-3:] /= np.linalg.norm(observed[-3:], axis=1, keepdims=True)
    observed = observed[rng.permutation(len(observed))]
    rotation, template_indices, observed_indices, residual = (
        estimate_sphere_template_pose(
            template, observed, inlier_chord_distance=.02,
            hypotheses=8000, seed=446))
    recovered, _ = cv2.Rodrigues(rotation)
    assert len(template_indices) >= 7
    assert np.allclose(recovered.reshape(3), expected_vector, atol=.01)
    assert residual < .01
    assert len(np.unique(observed_indices)) == len(observed_indices)


def test_geometric_hash_template_matcher_recovers_pose_with_fewer_hypotheses():
    rng = np.random.default_rng(447)
    template = rng.normal(size=(12, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    expected_vector = np.array([-.55, .35, .8])
    expected_rotation, _ = cv2.Rodrigues(expected_vector)
    retained = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    observed = (expected_rotation @ template[retained].T).T
    observed += rng.normal(0, .002, observed.shape)
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    observed = np.vstack((observed, rng.normal(size=(3, 3))))
    observed[-3:] /= np.linalg.norm(observed[-3:], axis=1, keepdims=True)
    observed = observed[rng.permutation(len(observed))]
    matcher = SphereTemplateMatcher(
        template, signature_tolerance=.025, inlier_chord_distance=.025,
        maximum_hypotheses=2000)
    rotation, template_indices, observed_indices, residual, tested = matcher.match(observed)
    recovered, _ = cv2.Rodrigues(rotation)
    assert len(template_indices) >= len(retained)
    assert np.allclose(recovered.reshape(3), expected_vector, atol=.015)
    assert residual < .012
    assert tested < 2000


def test_template_pose_sequence_recovers_constant_3d_spin_and_rpm():
    timestamps = np.arange(31, dtype=np.int64) * 1000
    expected = np.array([120.0, -55.0, 80.0])
    rotations = np.asarray([
        cv2.Rodrigues(expected * (timestamp * 1e-6))[0]
        for timestamp in timestamps
    ])
    # One bad absolute pose produces two bad adjacent pairs; robust fitting
    # must reject them without turning the jump into a plausible RPM.
    rotations[14] = cv2.Rodrigues(np.array([2.1, -.8, .4]))[0]
    estimate = estimate_spin_from_template_poses(timestamps, rotations)
    expected_rpm = np.linalg.norm(expected) * 60 / (2 * np.pi)
    assert estimate.accepted is True
    assert np.allclose(estimate.omega_rad_s, expected, atol=.5)
    assert abs(estimate.rpm - expected_rpm) < 5
    assert estimate.inlier_pairs == 28
    assert estimate.axis_resultant > .99


def test_template_pose_sequence_fails_closed_on_inconsistent_axes():
    timestamps = np.arange(12, dtype=np.int64) * 1000
    rotations = [np.eye(3)]
    for index in range(1, len(timestamps)):
        axis = np.eye(3)[index % 3]
        step = cv2.Rodrigues(axis * .12)[0]
        rotations.append(step @ rotations[-1])
    estimate = estimate_spin_from_template_poses(timestamps, np.asarray(rotations))
    assert estimate.accepted is False
    assert estimate.rpm is None


def test_template_pose_tracker_uses_prediction_after_global_acquisition():
    rng = np.random.default_rng(448)
    template = rng.normal(size=(14, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    tracker = SphereTemplatePoseTracker(
        template, signature_tolerance=.02, inlier_chord_distance=.025,
        minimum_markers=5, maximum_hypotheses=2500)
    omega = np.array([500.0, -220.0, 310.0])
    modes = []
    poses = []
    for index in range(12):
        truth = cv2.Rodrigues(omega * index * .001)[0]
        retained = np.arange(14)[(np.arange(14) + index) % 5 != 0]
        observed = (truth @ template[retained].T).T
        observed += rng.normal(0, .001, observed.shape)
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        pose, ids, _, residual, mode, _ = tracker.update(index * 1000, observed)
        assert len(ids) >= 5 and residual < .01
        modes.append(mode); poses.append(pose)
    assert modes[:2] == ["global", "global"]
    assert modes[2:].count("local") >= 9
    estimate = estimate_spin_from_template_poses(
        np.arange(12) * 1000, np.asarray(poses), minimum_pairs=6)
    assert estimate.accepted is True
    assert np.allclose(estimate.omega_rad_s, omega, atol=1.0)


def test_labeled_multiview_frames_build_connected_asymmetric_template():
    rng = np.random.default_rng(449)
    truth = rng.normal(size=(13, 3))
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    frames = []
    for index in range(18):
        pose = cv2.Rodrigues(np.array([.07, -.04, .09]) * index)[0]
        # Keep a connected core while rotating peripheral IDs in and out.
        retained = sorted(set([0, 1, 2, 3, 4]) |
                          {5 + ((index + offset) % 8) for offset in range(4)})
        observed = (pose @ truth[retained].T).T
        observed += rng.normal(0, .001, observed.shape)
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        frames.append({
            "timestamp_us": index * 1000,
            "directions": {identifier: value for identifier, value in zip(retained, observed)},
        })
    result = build_labeled_sphere_template(frames)
    assert result.accepted is True
    assert len(result.marker_ids) == 13
    assert len(result.timestamps_us) == 18
    expected_in_body = (cv2.Rodrigues(np.zeros(3))[0] @ truth.T).T
    recovered = result.directions[np.argsort(result.marker_ids)]
    assert np.max(np.linalg.norm(recovered - expected_in_body, axis=1)) < .01


def test_labeled_template_builder_rejects_disconnected_sparse_frames():
    frames = [
        {"timestamp_us": 0, "directions": {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}},
        {"timestamp_us": 1000,
         "directions": {3: [1, 0, 0], 4: [0, 1, 0], 5: [0, 0, 1]}},
    ]
    result = build_labeled_sphere_template(frames)
    assert result.accepted is False
    assert "requires at least" in result.reason


def test_template_symmetry_audit_accepts_asymmetric_random_constellation():
    rng = np.random.default_rng(450)
    template = rng.normal(size=(13, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    report = audit_sphere_template_symmetry(template, maximum_hypotheses=15000)
    assert report["passed"] is True
    assert report["maximum_nontrivial_match_fraction"] < .70


def test_template_symmetry_audit_rejects_octahedral_constellation():
    template = np.vstack((np.eye(3), -np.eye(3)))
    report = audit_sphere_template_symmetry(template, maximum_hypotheses=5000)
    assert report["passed"] is False
    assert report["maximum_nontrivial_match_fraction"] == 1.0


def test_streaming_template_spin_publishes_only_after_rolling_gate_passes():
    rng = np.random.default_rng(451)
    template = rng.normal(size=(14, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    omega = np.array([210.0, -95.0, 160.0])
    stream = CalibratedTemplateSpinStream(
        template, template_validated=True, window_us=20000,
        inlier_chord_distance=.025)
    results = []
    for index in range(20):
        rotation = cv2.Rodrigues(omega * index * .001)[0]
        observed = (rotation @ template.T).T
        observed += rng.normal(0, .0008, observed.shape)
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        results.append(stream.update(index * 1000, observed))
    assert all(item.rpm is None for item in results[:6])
    assert results[-1].rpm is not None
    expected_rpm = np.linalg.norm(omega) * 60 / (2 * np.pi)
    assert abs(results[-1].rpm - expected_rpm) < 2.0
    assert results[-1].pose_mode == "local"


def test_streaming_template_spin_never_publishes_from_unvalidated_template():
    template = np.eye(3)
    stream = CalibratedTemplateSpinStream(template, template_validated=False)
    for index in range(20):
        result = stream.update(index * 1000, template)
        assert result.rpm is None
        assert result.pose_mode == "disabled_unvalidated_template"


def test_streaming_template_spin_reuses_expensive_fit_between_publish_intervals():
    rng = np.random.default_rng(452)
    template = rng.normal(size=(14, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    omega = np.array([90.0, -40.0, 70.0])
    stream = CalibratedTemplateSpinStream(
        template, template_validated=True, window_us=20000,
        inlier_chord_distance=.025, spin_update_interval_us=5000)
    results = []
    for index in range(13):
        rotation = cv2.Rodrigues(omega * index * .001)[0]
        results.append(stream.update(index * 1000, (rotation @ template.T).T))
    assert results[6].spin is not None
    assert results[10].spin is results[6].spin
    assert results[11].spin is not results[10].spin
    assert results[11].rpm is not None


def test_pooled_constant_rotation_recovers_from_sparse_majority_outliers():
    rng = np.random.default_rng(445)
    count = 240
    source = rng.normal(size=(count, 3))
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    dt = rng.uniform(.0008, .003, count)
    expected = np.array([90., -55., 180.])
    target = _rotate_vectors(source, expected, dt)
    target += rng.normal(0, .002, target.shape)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    outliers = rng.choice(count, 150, replace=False)
    target[outliers] = rng.normal(size=(len(outliers), 3))
    target[outliers] /= np.linalg.norm(target[outliers], axis=1, keepdims=True)
    omega, inliers, rms, condition = solve_constant_sphere_rotation(
        source, target, dt, minimum_correspondences=50,
        angular_threshold_rad=.015, hypotheses=800)
    assert np.count_nonzero(inliers) >= 80
    assert np.allclose(omega, expected, atol=2.0)
    assert rms < .006
    assert condition < 3
