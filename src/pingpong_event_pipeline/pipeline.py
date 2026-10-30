from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import ClassicalBallDetector, HoughCircleBallDetector
from .events import (EVENT_DTYPE, IncrementalTimeSurface, RawWindowReader,
                     motion_compensated_ball_patch, render_bgr, synchronized_batches)
from .models import Detection2D
from .spin import ImagePlaneSpinEstimator, fuse_optical_axis_spin, solve_sphere_angular_velocity
from .stereo import StereoCalibration, select_stereo_detection_pair
from .tracking import BallTracker2D
from .trajectory import TrajectoryFilter3D


CSV_FIELDS = [
    "timestamp_us", "master_x", "master_y", "master_radius", "master_confidence",
    "slave_x", "slave_y", "slave_radius", "slave_confidence",
    "measured_x_m", "measured_y_m", "measured_z_m",
    "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "speed_mps",
    "reprojection_error_px", "master_spin_rpm", "slave_spin_rpm",
    "master_spin_features", "slave_spin_features", "master_spin_quality", "slave_spin_quality",
    "master_omega_x", "master_omega_y", "master_omega_z",
    "slave_omega_master_x", "slave_omega_master_y", "slave_omega_master_z",
    "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s", "spin_rpm",
    "spin_axis_x", "spin_axis_y", "spin_axis_z", "spin_quality", "spin_disagreement",
]


def _minimum_roi_event_count(master_events, slave_events, detector_cfg) -> int:
    roi = detector_cfg.get("search_roi")
    if roi is None:
        return min(len(master_events), len(slave_events))
    x0, y0, x1, y1 = map(int, roi)
    def count(events):
        if events is None or not len(events):
            return 0
        return int(np.count_nonzero((events["x"] >= x0) & (events["x"] < x1) &
                                    (events["y"] >= y0) & (events["y"] < y1)))
    return min(count(master_events), count(slave_events))


def _bootstrap_has_activity(master_events, slave_events, detector_cfg) -> bool:
    minimum = int(detector_cfg.get("bootstrap_min_roi_events", 0))
    return minimum <= 0 or _minimum_roi_event_count(
        master_events, slave_events, detector_cfg) >= minimum


def _draw(image, track, label, color):
    cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if track is None:
        return
    center = (round(track.x), round(track.y))
    cv2.circle(image, center, max(2, round(track.radius)), color, 2)
    cv2.arrowedLine(image, center,
                    (round(track.x + track.vx * 0.02), round(track.y + track.vy * 0.02)), color, 2)


def _detect_with_normalized_roi(accumulator, timestamp_us, tracker, detector, prediction,
                                event_cfg, spin_cfg, width, height):
    """Use a full frame once, then update the known ball in a tiny normalized ROI."""
    if prediction is None:
        search_roi = detector.cfg.get("search_roi")
        if search_roi is not None:
            x0, y0, x1, y1 = map(int, search_roi)
            surface = accumulator.region(
                search_roi, timestamp_us, event_cfg["sae_tau_us"], event_cfg["window_us"])
            local_cfg = dict(detector.cfg); local_cfg.pop("search_roi", None)
            local_cfg.update(detector.cfg.get("bootstrap_classical", {}))
            local_class = (ClassicalBallDetector if
                           detector.cfg.get("bootstrap_detector_type") == "classical" else
                           detector.__class__)
            local_detector = local_class(local_cfg)
            local = local_detector.detect(surface, timestamp_us, None)
            detections = [Detection2D(item.timestamp_us, item.x + x0, item.y + y0,
                                      item.radius, item.score, item.event_count) for item in local]
            if (detections and detector.cfg.get("bootstrap_refine_with_local_ring", False) and
                    local_class is ClassicalBallDetector):
                refined = []
                for coarse in detections[:int(detector.cfg.get("bootstrap_local_proposals", 1))]:
                    items, _ = _detect_with_normalized_roi(
                        accumulator, timestamp_us, tracker, detector,
                        np.array([coarse.x, coarse.y, coarse.radius], np.float64),
                        event_cfg, spin_cfg, width, height)
                    refined.extend(items)
                if refined:
                    return refined, None
            return detections, None
        surface = accumulator.surface(
            timestamp_us, event_cfg["sae_tau_us"], event_cfg["window_us"])
        return detector.detect(surface, timestamp_us, None), surface
    size = int(spin_cfg.get("normalized_patch_size", 64))
    radius = max(float(prediction[2]), 1.0)
    roi_scale = max(float(detector.cfg.get("normalized_roi_scale", 1.7)), 1.05)
    extent = radius * roi_scale
    patch = accumulator.patch(
        prediction[:2], extent, size, timestamp_us,
        event_cfg["sae_tau_us"], event_cfg["window_us"])
    half = 0.5 * size
    normalized_radius = half / roi_scale
    local = detector.detect(patch, timestamp_us, (half, half, normalized_radius))
    detections = [Detection2D(
        timestamp_us,
        float(prediction[0] + (item.x - half) * extent / half),
        float(prediction[1] + (item.y - half) * extent / half),
        float(item.radius * extent / half), item.score, item.event_count)
        for item in local]
    if detections or not bool(detector.cfg.get("predictive_fallback_hough", False)):
        return detections, None
    # A fast ball can leave the predicted ROI in one or two updates.  Without
    # this configured full-frame fallback the tracker keeps returning its old
    # state and can never reacquire the ball until it is reset.
    surface = accumulator.surface(
        timestamp_us, event_cfg["sae_tau_us"], event_cfg["window_us"])
    return detector.detect(surface, timestamp_us, None), surface


def _spin_channel(events, timestamp_us, track, estimator, spin_cfg):
    patch_size = int(spin_cfg.get("normalized_patch_size", 128))
    if track is None:
        return estimator.update(np.zeros((2, patch_size, patch_size), np.float32),
                                None, normalized_patch=True)
    surface = motion_compensated_ball_patch(
        events, track.point, track.radius, patch_size, timestamp_us,
        float(spin_cfg.get("sae_tau_us", 500)), (track.vx, track.vy),
        int(spin_cfg.get("window_us", 1000)))
    return estimator.update(surface, track, normalized_patch=True)


def run_offline(
    master_raw: str | Path,
    slave_raw: str | Path,
    config: dict,
    output_csv: str | Path,
    preview_path: Optional[str | Path] = None,
    calibration_path: Optional[str | Path] = None,
    max_windows: Optional[int] = None,
    start_offset_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> dict:
    # Small 128x128 ROIs are faster without OpenCV's default 8-thread dispatch.
    cv2.setNumThreads(int(config.get("runtime", {}).get("opencv_threads", 1)))
    master_reader, slave_reader = RawWindowReader(master_raw), RawWindowReader(slave_raw)
    width, height = config["cameras"]["width"], config["cameras"]["height"]
    event_cfg, detector_cfg = config["events"], config["detection"]
    detector_type = detector_cfg.get("type", "classical")
    detector_class = HoughCircleBallDetector if detector_type == "hough_circle" else ClassicalBallDetector
    master_detector = detector_class(detector_cfg)
    slave_detector = detector_class(detector_cfg)
    master_tracker = BallTracker2D(detector_cfg["max_missed_windows"])
    slave_tracker = BallTracker2D(detector_cfg["max_missed_windows"])
    calibration = StereoCalibration(calibration_path) if calibration_path and Path(calibration_path).exists() else None
    master_spin = ImagePlaneSpinEstimator(config["spin"], calibration.K1 if calibration else None)
    slave_spin = ImagePlaneSpinEstimator(config["spin"], calibration.K2 if calibration else None)
    trajectory = TrajectoryFilter3D(config["trajectory"]) if calibration else None
    master_accumulator = IncrementalTimeSurface(width, height)
    slave_accumulator = IncrementalTimeSurface(width, height)

    preview = None
    preview_every = max(1, round(1_000_000 / event_cfg["step_us"] / config["output"]["preview_fps"]))
    scale = float(config["output"]["preview_scale"])
    if preview_path:
        preview_size = (round(width * 2 * scale), round(height * scale))
        preview = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                  config["output"]["preview_fps"], preview_size)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = detections = stereo_states = master_spin_states = slave_spin_states = fused_spin_states = 0
    previous_measured_point = None
    previous_measured_timestamp = None
    consecutive_stereo_measurements = 0
    executor = ThreadPoolExecutor(max_workers=2) if config.get("runtime", {}).get(
        "parallel_cameras", True) else None
    # Load/compile JIT kernels before timing the persistent processing loop.
    if config["spin"]["enabled"]:
        dummy_events = np.zeros(1, dtype=EVENT_DTYPE)
        motion_compensated_ball_patch(dummy_events, (1.0, 1.0), 1.0, 8, 0, 1.0)
        warm_xy = np.array([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]])
        solve_sphere_angular_velocity(warm_xy, warm_xy + 1e-5, 0.001)
    processing_started = time.perf_counter()
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        corrections = config["cameras"]["timestamp_correction_us"]
        common_start = max(master_reader.info.start_us + corrections["master"],
                           slave_reader.info.start_us + corrections["slave"])
        range_start = common_start + round(max(start_offset_s, 0.0) * 1_000_000)
        range_end = (range_start + round(duration_s * 1_000_000)
                     if duration_s is not None else None)
        iterator = synchronized_batches(
            master_reader, slave_reader, event_cfg["step_us"],
            corrections["master"], corrections["slave"], range_start, range_end,
            warmup_us=event_cfg["window_us"],
        )
        master_spin_buffer = np.empty(0, dtype=EVENT_DTYPE)
        slave_spin_buffer = np.empty(0, dtype=EVENT_DTYPE)
        for index, (timestamp_us, master_events, slave_events) in enumerate(iterator):
            if max_windows is not None and index >= max_windows:
                break
            master_accumulator.ingest(master_events); slave_accumulator.ingest(slave_events)
            if config["spin"]["enabled"]:
                spin_age = int(config["spin"].get("window_us", event_cfg["window_us"]))
                master_spin_buffer = np.concatenate((master_spin_buffer, master_events))
                slave_spin_buffer = np.concatenate((slave_spin_buffer, slave_events))
                master_spin_buffer = master_spin_buffer[
                    master_spin_buffer["timestamp"] >= timestamp_us - spin_age]
                slave_spin_buffer = slave_spin_buffer[
                    slave_spin_buffer["timestamp"] >= timestamp_us - spin_age]
            master_prediction = master_tracker.x[[0, 1, 4]] if master_tracker.initialized else None
            slave_prediction = slave_tracker.x[[0, 1, 4]] if slave_tracker.initialized else None
            base_interval_us = int(detector_cfg.get("interval_us", event_cfg["step_us"]))
            settled_after = int(detector_cfg.get("settled_after_stereo_measurements", 0))
            detection_interval_us = (int(detector_cfg.get("settled_interval_us", base_interval_us))
                                     if settled_after > 0 and
                                     consecutive_stereo_measurements >= settled_after else
                                     base_interval_us)
            detection_stride = max(1, round(detection_interval_us / event_cfg["step_us"]))
            initialized = master_tracker.initialized and slave_tracker.initialized
            if initialized:
                run_detection = index % detection_stride == 0
            else:
                bootstrap_interval = int(detector_cfg.get("bootstrap_interval_us", detection_interval_us))
                bootstrap_stride = max(1, round(bootstrap_interval / event_cfg["step_us"]))
                activity = _minimum_roi_event_count(master_events, slave_events, detector_cfg)
                burst = activity >= int(detector_cfg.get("bootstrap_burst_roi_events", 2**31 - 1))
                run_detection = ((burst or index % bootstrap_stride == 0) and
                                 activity >= int(detector_cfg.get("bootstrap_min_roi_events", 0)))
            master_surface = slave_surface = None
            if run_detection:
                detect_args = [
                    (master_accumulator, timestamp_us, master_tracker, master_detector,
                     master_prediction, event_cfg, config["spin"], width, height),
                    (slave_accumulator, timestamp_us, slave_tracker, slave_detector,
                     slave_prediction, event_cfg, config["spin"], width, height),
                ]
                if executor:
                    future_master = executor.submit(_detect_with_normalized_roi, *detect_args[0])
                    future_slave = executor.submit(_detect_with_normalized_roi, *detect_args[1])
                    master_candidates, master_surface = future_master.result()
                    slave_candidates, slave_surface = future_slave.result()
                else:
                    master_candidates, master_surface = _detect_with_normalized_roi(*detect_args[0])
                    slave_candidates, slave_surface = _detect_with_normalized_roi(*detect_args[1])
            else:
                master_candidates, slave_candidates = [], []
            paired = None
            if calibration:
                # Avoid seeding both Kalman filters from weak partial arcs at
                # the image border. A few milliseconds of acquisition latency
                # prevents a false pair from poisoning the complete flight.
                if not master_tracker.initialized or not slave_tracker.initialized:
                    bootstrap_score = float(detector_cfg.get("bootstrap_min_score", 0.0))
                    bootstrap_radius = float(detector_cfg.get("bootstrap_min_radius_px", 0.0))
                    master_candidates = [d for d in master_candidates
                                         if d.score >= bootstrap_score and d.radius >= bootstrap_radius]
                    slave_candidates = [d for d in slave_candidates
                                        if d.score >= bootstrap_score and d.radius >= bootstrap_radius]
                previous_for_gate = previous_measured_point
                jump_limit = None
                if previous_measured_timestamp is not None:
                    gap_s = (timestamp_us - previous_measured_timestamp) * 1e-6
                    if gap_s > config["stereo"].get("continuity_reset_s", 0.05):
                        previous_for_gate = None
                    else:
                        jump_limit = (config["stereo"].get("maximum_ball_speed_mps", 30.0) * gap_s +
                                      config["stereo"].get("position_jump_allowance_m", 0.02))
                paired = select_stereo_detection_pair(
                    master_candidates, slave_candidates, calibration,
                    (config["stereo"]["max_epipolar_error_px"] if
                     master_tracker.initialized and slave_tracker.initialized else
                     detector_cfg.get("bootstrap_max_epipolar_error_px",
                                      config["stereo"]["max_epipolar_error_px"])),
                    config["stereo"].get("max_radius_ratio", 2.0),
                    config["stereo"].get("ball_radius_m"),
                    config["stereo"].get("minimum_depth_m", 0.3),
                    config["stereo"].get("maximum_depth_m", 3.0),
                    config["stereo"].get("max_radius_scale_error", 0.8),
                    master_prediction[:2] if master_prediction is not None else None,
                    slave_prediction[:2] if slave_prediction is not None else None,
                    config["stereo"].get("maximum_prediction_error_px", 60.0),
                    previous_for_gate, jump_limit,
                )
                master_candidates = [paired[0]] if paired else []
                slave_candidates = [paired[1]] if paired else []
            if run_detection:
                master_track = master_tracker.update(timestamp_us, master_candidates)
                slave_track = slave_tracker.update(timestamp_us, slave_candidates)
            else:
                master_track = master_tracker.advance(timestamp_us)
                slave_track = slave_tracker.advance(timestamp_us)
            detections += int(master_track is not None) + int(slave_track is not None)
            if config["spin"]["enabled"]:
                if executor:
                    future_master = executor.submit(
                        _spin_channel, master_spin_buffer, timestamp_us, master_track,
                        master_spin, config["spin"])
                    future_slave = executor.submit(
                        _spin_channel, slave_spin_buffer, timestamp_us, slave_track,
                        slave_spin, config["spin"])
                    master_spin_state, slave_spin_state = future_master.result(), future_slave.result()
                else:
                    master_spin_state = _spin_channel(
                        master_spin_buffer, timestamp_us, master_track, master_spin, config["spin"])
                    slave_spin_state = _spin_channel(
                        slave_spin_buffer, timestamp_us, slave_track, slave_spin, config["spin"])
            else:
                master_spin_state = slave_spin_state = None
            row = {field: "" for field in CSV_FIELDS}
            row["timestamp_us"] = timestamp_us
            for prefix, track in (("master", master_track), ("slave", slave_track)):
                if track:
                    row.update({f"{prefix}_x": track.x, f"{prefix}_y": track.y,
                                f"{prefix}_radius": track.radius, f"{prefix}_confidence": track.confidence})
            if master_spin_state:
                row.update({"master_spin_rpm": master_spin_state.rpm,
                            "master_spin_features": master_spin_state.feature_count,
                            "master_spin_quality": master_spin_state.quality,
                            "master_omega_x": master_spin_state.omega_camera_rad_s[0],
                            "master_omega_y": master_spin_state.omega_camera_rad_s[1],
                            "master_omega_z": master_spin_state.omega_camera_rad_s[2]})
                master_spin_states += 1
            if slave_spin_state:
                slave_in_master = (calibration.R.T @ slave_spin_state.omega_camera_rad_s
                                   if calibration else slave_spin_state.omega_camera_rad_s)
                row.update({"slave_spin_rpm": slave_spin_state.rpm,
                            "slave_spin_features": slave_spin_state.feature_count,
                            "slave_spin_quality": slave_spin_state.quality,
                            "slave_omega_master_x": slave_in_master[0],
                            "slave_omega_master_y": slave_in_master[1],
                            "slave_omega_master_z": slave_in_master[2]})
                slave_spin_states += 1

            if paired:
                point, reprojection = calibration.triangulate(paired[0].point, paired[1].point)
                if np.isfinite(point).all() and reprojection <= config["stereo"]["max_reprojection_error_px"]:
                    consecutive_stereo_measurements += 1
                    previous_measured_point = point.copy()
                    previous_measured_timestamp = timestamp_us
                    position, velocity = trajectory.update(timestamp_us, point)
                    row.update({"measured_x_m": point[0], "measured_y_m": point[1],
                                "measured_z_m": point[2],
                                "x_m": position[0], "y_m": position[1], "z_m": position[2],
                                "vx_mps": velocity[0], "vy_mps": velocity[1], "vz_mps": velocity[2],
                                "speed_mps": np.linalg.norm(velocity), "reprojection_error_px": reprojection})
                    stereo_states += 1
                elif run_detection:
                    consecutive_stereo_measurements = max(0, consecutive_stereo_measurements - 2)
            elif run_detection:
                consecutive_stereo_measurements = max(0, consecutive_stereo_measurements - 2)
            # Spin fusion is independent of whether this high-rate step also ran
            # the slower stereo circle detector.
            if calibration and master_spin_state and slave_spin_state:
                omega, quality = fuse_optical_axis_spin(
                    master_spin_state.omega_camera_rad_s, slave_spin_state.omega_camera_rad_s,
                    calibration.R.T, master_spin_state.quality, slave_spin_state.quality)
                magnitude = float(np.linalg.norm(omega))
                left_omega = master_spin_state.omega_camera_rad_s
                right_omega_in_left = calibration.R.T @ slave_spin_state.omega_camera_rad_s
                disagreement = float(np.linalg.norm(left_omega - right_omega_in_left) /
                                     max(magnitude, 1.0))
                row["spin_disagreement"] = disagreement
                accepted = (quality >= config["spin"].get("minimum_fused_quality", 0.15) and
                            disagreement <= config["spin"].get("maximum_relative_disagreement", 0.5))
                if accepted:
                    axis = omega / magnitude if magnitude > 1e-9 else np.zeros(3)
                    row.update({"omega_x_rad_s": omega[0], "omega_y_rad_s": omega[1],
                                "omega_z_rad_s": omega[2],
                                "spin_rpm": magnitude * 60.0 / (2.0 * np.pi),
                                "spin_axis_x": axis[0], "spin_axis_y": axis[1],
                                "spin_axis_z": axis[2], "spin_quality": quality})
                    fused_spin_states += 1
            writer.writerow(row)
            rows += 1

            if preview is not None and index % preview_every == 0:
                if master_surface is None:
                    master_surface = master_accumulator.surface(
                        timestamp_us, event_cfg["sae_tau_us"], event_cfg["window_us"])
                if slave_surface is None:
                    slave_surface = slave_accumulator.surface(
                        timestamp_us, event_cfg["sae_tau_us"], event_cfg["window_us"])
                left_image, right_image = render_bgr(master_surface), render_bgr(slave_surface)
                _draw(left_image, master_track, "MASTER F59", (0, 255, 0))
                _draw(right_image, slave_track, "SLAVE F3E", (0, 255, 0))
                composite = np.hstack([left_image, right_image])
                composite = cv2.resize(composite, None, fx=scale, fy=scale)
                preview.write(composite)
    if preview is not None:
        preview.release()
    if executor is not None:
        executor.shutdown(wait=True)
    processing_wall_s = time.perf_counter() - processing_started
    processed_data_s = rows * event_cfg["step_us"] * 1e-6
    return {"windows": rows, "tracked_2d_states": detections, "stereo_3d_states": stereo_states,
            "master_spin_states": master_spin_states, "slave_spin_states": slave_spin_states,
            "fused_spin_states": fused_spin_states,
            "processing_wall_s": processing_wall_s,
            "processed_data_s": processed_data_s,
            "realtime_factor": processed_data_s / max(processing_wall_s, 1e-9),
            "csv": str(output_csv.resolve()), "preview": str(Path(preview_path).resolve()) if preview_path else None}
