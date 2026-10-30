import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pingpong_event_pipeline.events import EVENT_DTYPE
from pingpong_event_pipeline.models import Track2D
from pingpong_event_pipeline.stereo import StereoCalibration
from pingpong_event_pipeline.template_spin_channel import StereoTemplateSpinChannel


ROOT = Path(__file__).parents[1]
CALIBRATION = ROOT / "calibration" / "stereo_calibration.npz"


def _project(calibration, points, camera):
    if camera == 0:
        rvec = np.zeros(3); tvec = np.zeros(3)
        K, D = calibration.K1, calibration.D1
    else:
        rvec, _ = cv2.Rodrigues(calibration.R)
        tvec = calibration.T.reshape(3)
        K, D = calibration.K2, calibration.D2
    projected, _ = cv2.projectPoints(np.asarray(points), rvec, tvec, K, D)
    return projected.reshape(-1, 2)


def _events(points, timestamp):
    samples = []
    for point in points:
        cx, cy = np.rint(point).astype(int)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                samples.append((cx + dx, cy + dy, 1, timestamp))
    valid = [item for item in samples if 0 <= item[0] < 1280 and 0 <= item[1] < 720]
    result = np.empty(len(valid), dtype=EVENT_DTYPE)
    if valid:
        values = np.asarray(valid, np.uint64)
        result["x"] = values[:, 0]; result["y"] = values[:, 1]
        result["polarity"] = values[:, 2]; result["timestamp"] = values[:, 3]
    return result


@pytest.mark.skipif(not CALIBRATION.is_file(), reason="real stereo calibration is an external local asset")
def test_events_to_stereo_markers_to_streaming_rpm_with_real_calibration(tmp_path):
    calibration = StereoCalibration(CALIBRATION)
    rng = np.random.default_rng(1201)
    template = []
    while len(template) < 42:
        candidate = rng.normal(size=3); candidate /= np.linalg.norm(candidate)
        if all(np.linalg.norm(candidate - old) > .12 for old in template):
            template.append(candidate)
    template = np.asarray(template)
    template_path = tmp_path / "template.npz"
    np.savez_compressed(template_path, directions=template)
    report_path = tmp_path / "template.json"
    report_path.write_text(json.dumps({
        "physical_template_geometry_accepted": True,
        "candidate_tracking_identity_accepted": True,
        "spin_solver_allowed": True,
    }), encoding="utf-8")
    channel = StereoTemplateSpinChannel(
        CALIBRATION, template_path, report_path, {
            "window_us": 400, "tau_us": 200, "patch_size": 64,
            "normalized_patch_size": 64, "dot_mask_radius_fraction": .98,
            "dot_blob_close_px": 1, "dot_blob_min_area_px": 3,
            "dot_blob_max_area_px": 180, "maximum_dots": 42,
            "template_inlier_chord": .20,
            "signature_tolerance": .15,
            "guided_marker_gate_px": 6.0,
            "maximum_reprojection_px": 1.2,
            "maximum_shell_error_m": .003,
        })
    center = np.array([0.0, 0.0, 1.0])
    radius_m = .020
    centers_px = [_project(calibration, center[None], camera)[0] for camera in (0, 1)]
    center_camera = [center, calibration.R @ center + calibration.T.reshape(3)]
    radii_px = [calibration.K1[0, 0] * radius_m / center_camera[0][2],
                calibration.K2[0, 0] * radius_m / center_camera[1][2]]
    omega = np.array([72.0, -38.0, 55.0])
    results = []
    base = 1_000_000
    for index in range(24):
        timestamp = base + index * 1000
        rotation = cv2.Rodrigues(omega * index * .001)[0]
        directions = (rotation @ template.T).T
        points_3d = center + radius_m * directions
        projected = [_project(calibration, points_3d, camera) for camera in (0, 1)]
        visible = np.ones(len(template), bool)
        for camera in (0, 1):
            camera_points = (points_3d if camera == 0 else
                             (calibration.R @ points_3d.T).T + calibration.T.reshape(1, 3))
            normals = directions if camera == 0 else (calibration.R @ directions.T).T
            facing = np.sum(normals * (-camera_points), axis=1) > 0
            normalized_radius = np.linalg.norm(
                projected[camera] - centers_px[camera], axis=1) / radii_px[camera]
            visible &= facing & (normalized_radius < .94)
        master_events = _events(projected[0][visible], timestamp)
        slave_events = _events(projected[1][visible], timestamp)
        tracks = [Track2D(timestamp, centers_px[camera][0], centers_px[camera][1],
                          0.0, 0.0, radii_px[camera], 1.0, True)
                  for camera in (0, 1)]
        results.append(channel.process(
            timestamp, master_events, slave_events, tracks[0], tracks[1], center))
    valid = [item for item in results if item.rpm is not None]
    assert valid, [(item.master_candidates, item.slave_candidates,
                    item.triangulated_markers,
                    item.stream.template_inliers if item.stream else 0,
                    item.stream.pose_mode if item.stream else None,
                    item.stream.window_pose_frames if item.stream else 0,
                    item.stream.spin.reason if item.stream and item.stream.spin else None)
                   for item in results]
    expected_rpm = np.linalg.norm(omega) * 60 / (2 * np.pi)
    assert abs(valid[-1].rpm - expected_rpm) < 30.0
    expected_axis = omega / np.linalg.norm(omega)
    axis_error = np.arccos(np.clip(
        np.dot(valid[-1].stream.spin.axis, expected_axis), -1.0, 1.0))
    assert axis_error < .06
    assert valid[-1].triangulated_markers >= 4
    assert valid[-1].stream.pose_mode == "guided"
