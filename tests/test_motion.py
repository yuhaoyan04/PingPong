import csv
import json

import cv2
import numpy as np
import pytest

from pingpong_event_pipeline.motion import ball_motion, local_velocity, stereo_motion


def test_physical_microseconds_and_time_scale_are_used():
    t = 650_000_000 + np.arange(21) * 500
    seconds = (t - t[0]) * 1e-6
    p = np.c_[10 + 2000 * seconds, 50 - 3000 * seconds]
    velocity = local_velocity(t, p)
    np.testing.assert_allclose(velocity, np.tile([2000, -3000], (len(t), 1)), atol=1e-6)
    slower = local_velocity(2 * t, p, max_gap_us=1500)
    np.testing.assert_allclose(slower, velocity / 2, atol=1e-6)


def test_invalid_points_and_gaps_do_not_fabricate_velocity():
    t = np.array([0, 500, 1000, 1500, 2000, 2500, 5000, 5500, 6000])
    p = np.c_[t * .001, np.zeros(len(t))]
    p[3] = 1e8
    valid = np.ones(len(t), bool); valid[3] = False
    velocity = local_velocity(t, p, valid)
    assert np.isnan(velocity[3:6]).all()  # invalid sample + only two remaining observations
    np.testing.assert_allclose(velocity[:3, 0], 1000, atol=1e-6)
    np.testing.assert_allclose(velocity[6:, 0], 1000, atol=1e-6)


def test_bad_timebase_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        local_velocity([1, 1, 2], np.zeros((3, 2)))


def test_ball_diameter_scaling_is_only_a_conditional_transverse_speed(tmp_path):
    folder = tmp_path / "unlabeled_inference/F3E/serve02_F3E"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"camera": "F3E"}))
    rows = [dict(image=f"{i}.png", frame_index=i, timestamp_us=500 * i,
                 center_x_px=50 + i, center_y_px=30, radius_px=20, track_support="ring_supported") for i in range(7)]
    with (folder / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result, groups = ball_motion(tmp_path, ball_diameter_m=.040)
    assert groups[0]["image_speed_median_px_s"] == pytest.approx(2000)
    assert result[3]["candidate_transverse_speed_mps"] == pytest.approx(2)
    assert result[3]["projected_speed_ball_diameters_s"] == pytest.approx(50)
    assert result[3]["speed_mps"] is None
    assert not result[3]["physical_measurement_validated"]
    larger, _ = ball_motion(tmp_path, ball_diameter_m=.080)
    assert larger[3]["candidate_transverse_speed_mps"] == pytest.approx(4)


def calibration_fixture(tmp_path):
    K = np.array([[600., 0, 640], [0, 600, 360], [0, 0, 1]])
    D, R, T = np.zeros(5), np.eye(3), np.array([-.1, 0, 0])
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K, D, K, D, (1280, 720), R, T.reshape(3, 1))
    tx = np.array([[0, 0, 0], [0, 0, .1], [0, -.1, 0]])
    F = np.linalg.inv(K).T @ tx @ np.linalg.inv(K)
    path = tmp_path / "synthetic_calibration.npz"
    np.savez(path, K1=K, K2=K, D1=D, D2=D, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, F=F)
    return path, K, T


def test_actual_calibration_interface_recovers_known_synthetic_3d_speed(tmp_path):
    path, K, T = calibration_fixture(tmp_path)
    rows, truth = [], []
    for i in range(15):
        t = i * .0005
        point = np.array([.01 + 2*t, .03 + t, 1 + .5*t]); truth.append(point)
        for camera, shift in [("F59", np.zeros(3)), ("F3E", T)]:
            projected = K @ (point + shift); projected = projected[:2] / projected[2]
            rows.append(dict(camera=camera, group="serve02_" + camera, timestamp_us=1_000_000 + i*500,
                             frame_index=i, center_x_px=projected[0], center_y_px=projected[1],
                             track_support="ring_supported"))
    output = stereo_motion(rows, path, "F59")
    np.testing.assert_allclose([[r["x_m"], r["y_m"], r["z_m"]] for r in output], truth, atol=1e-10)
    np.testing.assert_allclose([r["candidate_3d_speed_mps"] for r in output], np.linalg.norm([2, 1, .5]), atol=1e-6)
    assert all(r["speed_mps"] is None and not r["physical_measurement_validated"] for r in output)
    # Timestamp correction must be applied in its declared direction.
    for r in rows:
        if r["camera"] == "F3E": r["timestamp_us"] += 500
    corrected = stereo_motion(rows, path, "F59", slave_offset_us=-500)
    np.testing.assert_allclose([r["candidate_3d_speed_mps"] for r in corrected], np.linalg.norm([2, 1, .5]), atol=1e-6)


def test_stereo_calibration_requires_explicit_camera_mapping(tmp_path):
    with pytest.raises(ValueError, match="master"):
        stereo_motion([], tmp_path / "missing.npz", None)
