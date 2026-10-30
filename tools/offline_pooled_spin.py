"""Estimate one constant 3-D spin from pooled stereo event-patch optical flow."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import (
    RawWindowReader,
    motion_compensated_ball_patch,
    trajectory_compensated_ball_patch,
)
from pingpong_event_pipeline.spin import image_points_to_unit_sphere, solve_constant_sphere_rotation
from pingpong_event_pipeline.stereo import StereoCalibration


def patch_to_image(points, center, radius, size):
    return ((np.asarray(points, np.float64) - size / 2) * (float(radius) / (size / 2)) +
            np.asarray(center, np.float64))


def compact_texture_image(patch: np.ndarray) -> np.ndarray:
    """Suppress the broad translating-ball crescent before feature tracking."""
    activity = np.maximum(patch[0], patch[1]).astype(np.float32)
    fine = cv2.GaussianBlur(activity, (0, 0), 1.2)
    broad = cv2.GaussianBlur(activity, (0, 0), 8.0)
    residual = np.maximum(fine - 1.08 * broad, 0.0)
    positive = residual[residual > 0]
    scale = float(np.percentile(positive, 99.5)) if len(positive) else 0.0
    if scale <= 1e-7:
        return np.zeros(activity.shape, np.uint8)
    return np.clip(residual * (255.0 / scale), 0, 255).astype(np.uint8)


def prepare_camera_sequence(reader, path, window_us, tau_us, size, first, last,
                            max_features=None, exact_trajectory=False,
                            suppress_broad=False):
    """Render each motion-compensated event image once for all temporal baselines."""
    timestamps = path["timestamps_us"].astype(np.int64)
    centers = path["centers_px"].astype(np.float64)
    radii = path["radii_px"].astype(np.float64)
    dt_path = np.gradient(timestamps).astype(np.float64) * 1e-6
    velocities = np.column_stack([
        np.gradient(centers[:, axis]) / dt_path for axis in range(2)])
    events = reader.window(int(timestamps[first] - window_us), int(timestamps[last]))
    event_times = events["timestamp"].astype(np.int64)
    images = []
    indices = []
    for i in range(first, last + 1):
        timestamp = int(timestamps[i])
        lo = np.searchsorted(event_times, timestamp - window_us, side="right")
        hi = np.searchsorted(event_times, timestamp, side="right")
        if exact_trajectory:
            patch = trajectory_compensated_ball_patch(
                events[lo:hi], timestamps, centers, radii, size, timestamp,
                tau_us, window_us)
        else:
            patch = motion_compensated_ball_patch(
                events[lo:hi], centers[i], radii[i], size, timestamp, tau_us,
                velocities[i], window_us)
        image = (compact_texture_image(patch) if suppress_broad else
                 np.clip(np.maximum(patch[0], patch[1]) * 255, 0, 255).astype(np.uint8))
        images.append(image)
        indices.append(i)
    features = None
    if max_features is not None:
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(mask, (size // 2, size // 2), round(size * .38), 255, -1)
        features = [cv2.goodFeaturesToTrack(
            image, maxCorners=max_features, qualityLevel=.015, minDistance=2,
            mask=mask, blockSize=3) for image in images]
    return {
        "timestamps": timestamps, "centers": centers, "radii": radii,
        "images": images, "indices": np.asarray(indices, np.int32), "size": size,
        "features": features,
    }


def collect_camera_from_sequence(sequence, camera_matrix, rotation_to_master,
                                 lag_frames, max_features):
    timestamps = sequence["timestamps"]
    centers = sequence["centers"]
    radii = sequence["radii"]
    images = sequence["images"]
    indices = sequence["indices"]
    size = sequence["size"]
    sphere0 = []; sphere1 = []; delta_t = []; frame_ids = []
    features = sequence.get("features")
    mask = None
    if features is None:
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(mask, (size // 2, size // 2), round(size * .38), 255, -1)
    for position in range(lag_frames, len(images)):
        image = images[position]
        old_image = images[position - lag_frames]
        i = int(indices[position])
        old_i = int(indices[position - lag_frames])
        points = (features[position - lag_frames] if features is not None else
                  cv2.goodFeaturesToTrack(old_image, maxCorners=max_features,
                                          qualityLevel=.015, minDistance=2,
                                          mask=mask, blockSize=3))
        if points is not None and len(points) >= 4:
            new, status, error = cv2.calcOpticalFlowPyrLK(
                old_image, image, points, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
            if new is None:
                continue
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                image, old_image, new, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
            if back is not None:
                p0 = points.reshape(-1, 2); p1 = new.reshape(-1, 2)
                fb = np.linalg.norm(back.reshape(-1, 2) - p0, axis=1)
                good = (status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool) &
                        (error.reshape(-1) <= 40) & (fb <= 1.8))
                p0, p1 = p0[good], p1[good]
                inside = ((np.linalg.norm(p0 - size / 2, axis=1) <= size * .39) &
                          (np.linalg.norm(p1 - size / 2, axis=1) <= size * .43))
                p0, p1 = p0[inside], p1[inside]
                if len(p0):
                    image0 = patch_to_image(p0, centers[old_i], radii[old_i], size)
                    image1 = patch_to_image(p1, centers[i], radii[i], size)
                    s0 = image_points_to_unit_sphere(
                        image0, centers[old_i], radii[old_i], camera_matrix)
                    s1 = image_points_to_unit_sphere(
                        image1, centers[i], radii[i], camera_matrix)
                    s0 = (rotation_to_master @ s0.T).T
                    s1 = (rotation_to_master @ s1.T).T
                    sphere0.append(s0); sphere1.append(s1)
                    delta_t.append(np.full(len(s0), (timestamps[i] - timestamps[old_i]) * 1e-6))
                    frame_ids.append(np.full(len(s0), i, np.int32))
    empty3 = np.empty((0, 3), np.float64)
    return (np.vstack(sphere0) if sphere0 else empty3,
            np.vstack(sphere1) if sphere1 else empty3.copy(),
            np.concatenate(delta_t) if delta_t else np.empty(0),
            np.concatenate(frame_ids) if frame_ids else np.empty(0, np.int32))


def collect_camera(reader, path, camera_matrix, rotation_to_master, window_us, tau_us,
                   size, first, last, lag_frames, max_features):
    """Backward-compatible single-baseline collection entry point."""
    sequence = prepare_camera_sequence(
        reader, path, window_us, tau_us, size, first, last, max_features)
    return collect_camera_from_sequence(
        sequence, camera_matrix, rotation_to_master, lag_frames, max_features)


def solve(s0, s1, dt, threshold, max_rpm, seed, hypotheses=800,
          evaluation_limit=2500, finalist_count=12):
    omega, inliers, rms, condition = solve_constant_sphere_rotation(
        s0, s1, dt, minimum_correspondences=max(20, min(80, len(s0) // 8)),
        angular_threshold_rad=threshold, hypotheses=hypotheses,
        maximum_rpm=max_rpm, seed=seed,
        ransac_evaluation_limit=evaluation_limit, finalist_count=finalist_count)
    magnitude = float(np.linalg.norm(omega))
    return omega, inliers, {
        "correspondences": len(s0), "inliers": int(np.count_nonzero(inliers)),
        "inlier_fraction": float(np.mean(inliers)) if len(inliers) else 0.0,
        "angular_rms_deg": float(np.degrees(rms)) if np.isfinite(rms) else None,
        "condition": float(condition) if np.isfinite(condition) else None,
        "rpm": magnitude * 60 / (2 * np.pi) if magnitude else None,
        "axis_master": (omega / magnitude).tolist() if magnitude else None,
    }


def angular_threshold_for_lag(base_threshold, lag_frames, scale):
    """Allow endpoint LK uncertainty to grow linearly with temporal baseline."""
    return base_threshold * (lag_frames + 3) / 4 if scale else base_threshold


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-raw", required=True); parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--paths", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    parser.add_argument("--window-us", type=int, default=1000)
    parser.add_argument("--tau-us", type=int, default=500)
    parser.add_argument("--first-index", type=int); parser.add_argument("--last-index", type=int)
    parser.add_argument("--angular-threshold", type=float, default=.01)
    parser.add_argument("--scale-angular-threshold", action="store_true",
                        help="Scale threshold by (lag + 3) / 4 for multi-baseline LK error")
    parser.add_argument("--maximum-rpm", type=float, default=12000)
    parser.add_argument("--lag-frames", type=int, nargs="+", default=[1],
                        help="One or more temporal baselines; images are shared between them")
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--exact-trajectory-compensation", action="store_true",
                        help="Warp every event with its interpolated trajectory sample")
    parser.add_argument("--suppress-broad-structure", action="store_true",
                        help="Track compact internal texture after removing the broad ball crescent")
    parser.add_argument("--fast", action="store_true",
                        help="Use subset scoring for a quick provisional estimate")
    parser.add_argument("--hypotheses", type=int)
    parser.add_argument("--camera-hypotheses", type=int)
    parser.add_argument("--ransac-evaluation-limit", type=int)
    parser.add_argument("--ransac-finalists", type=int, default=12)
    args = parser.parse_args()
    hypotheses = args.hypotheses if args.hypotheses is not None else (800 if args.fast else 1200)
    camera_hypotheses = (args.camera_hypotheses if args.camera_hypotheses is not None
                         else (300 if args.fast else 1200))
    evaluation_limit = (args.ransac_evaluation_limit
                        if args.ransac_evaluation_limit is not None else (2500 if args.fast else 0))
    calibration = StereoCalibration(args.calibration); path_dir = Path(args.paths)
    paths = [np.load(path_dir / "master_path.npz"), np.load(path_dir / "slave_path.npz")]
    selected = paths[0]["selected"] if "selected" in paths[0] else np.zeros(len(paths[0]["timestamps_us"]))
    valid_indices = np.flatnonzero(selected >= 0)
    first = args.first_index if args.first_index is not None else int(valid_indices[0])
    last = args.last_index if args.last_index is not None else int(valid_indices[-1])
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    render_started = time.perf_counter()
    camera_matrices = (calibration.K1, calibration.K2)
    rotations = (np.eye(3), calibration.R.T)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(
            prepare_camera_sequence, reader, path, args.window_us, args.tau_us,
            128, first, last, max(8, args.max_features),
            args.exact_trajectory_compensation, args.suppress_broad_structure)
            for reader, path in zip(readers, paths)]
        sequences = [future.result() for future in futures]
    render_finished = time.perf_counter()

    lags = sorted(set(max(1, value) for value in args.lag_frames))
    requested_output = Path(args.output); requested_output.parent.mkdir(parents=True, exist_ok=True)
    collection_jobs = [(lag, camera) for lag in lags for camera in range(2)]
    with ThreadPoolExecutor(max_workers=min(8, len(collection_jobs))) as executor:
        futures = [executor.submit(
            collect_camera_from_sequence, sequences[camera], camera_matrices[camera],
            rotations[camera], lag, max(8, args.max_features))
            for lag, camera in collection_jobs]
        collected = {job: future.result() for job, future in zip(collection_jobs, futures)}
    collection_finished = time.perf_counter()
    lag_data = {}
    solve_jobs = []
    for lag in lags:
        data = [collected[(lag, camera)] for camera in range(2)]
        combined = tuple(np.concatenate([data[0][i], data[1][i]]) for i in range(4))
        angular_threshold = angular_threshold_for_lag(
            args.angular_threshold, lag, args.scale_angular_threshold)
        lag_data[lag] = (data, combined, angular_threshold)
        solve_jobs.extend([
            (lag, "combined", (*combined[:3], angular_threshold, args.maximum_rpm, 101,
                               max(100, hypotheses), evaluation_limit, args.ransac_finalists)),
            (lag, "master", (*data[0][:3], angular_threshold, args.maximum_rpm, 201,
                             max(100, camera_hypotheses), evaluation_limit,
                             args.ransac_finalists)),
            (lag, "slave", (*data[1][:3], angular_threshold, args.maximum_rpm, 202,
                            max(100, camera_hypotheses), evaluation_limit,
                            args.ransac_finalists)),
        ])
    with ThreadPoolExecutor(max_workers=min(8, len(solve_jobs))) as executor:
        futures = [executor.submit(solve, *values) for _, _, values in solve_jobs]
        solved_by_lag = {(lag, kind): future.result()
                         for (lag, kind, _), future in zip(solve_jobs, futures)}
    solve_finished = time.perf_counter()
    output_reports = []
    for lag in lags:
        data, combined, angular_threshold = lag_data[lag]
        solved = [solved_by_lag[(lag, kind)] for kind in ("combined", "master", "slave")]
        omega, inliers, report = solved[0]
        report.update({"first_index": first, "last_index": last, "lag_frames": lag,
                       "angular_threshold_rad": angular_threshold,
                       "duration_ms": (int(paths[0]["timestamps_us"][last]) -
                                       int(paths[0]["timestamps_us"][first])) * 1e-3})
        camera_omegas = [solved[1][0], solved[2][0]]
        report["master"] = solved[1][2]
        report["slave_in_master"] = solved[2][2]
        if all(np.linalg.norm(value) > 0 for value in camera_omegas):
            a, b = camera_omegas
            report["camera_axis_angle_deg"] = float(np.degrees(np.arccos(np.clip(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
            report["camera_relative_rpm_difference"] = float(
                abs(np.linalg.norm(a) - np.linalg.norm(b)) / max(np.linalg.norm(omega), 1e-9))
        output = (requested_output if len(lags) == 1 else
                  requested_output.with_name(f"spin_lag{lag}.json"))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez(output.with_suffix(".npz"), sphere0=combined[0], sphere1=combined[1],
                 dt_s=combined[2], frame_ids=combined[3], inliers=inliers,
                 omega_rad_s=omega)
        output_reports.append({"lag_frames": lag, "output": str(output), "report": report})
        print(json.dumps(report, indent=2))
    if len(lags) > 1:
        runtime = {
            "render_seconds": render_finished - render_started,
            "correspondence_seconds": collection_finished - render_finished,
            "solve_seconds": solve_finished - collection_finished,
            "total_seconds": time.perf_counter() - started,
        }
        requested_output.write_text(
            json.dumps({"runtime": runtime, "results": output_reports}, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
