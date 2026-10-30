"""Audit 3-D spin from stereo marker constellations in recorded RAW files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, motion_compensated_ball_patch, render_bgr
from pingpong_event_pipeline.spin import ImagePlaneSpinEstimator, _rotation_vector
from pingpong_event_pipeline.stereo import StereoCalibration
from pingpong_event_pipeline.stereo_spin import (
    estimate_unordered_sphere_rotation,
    triangulate_sphere_markers,
)


def parse_point(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], np.float64)
    if values.shape != (2,):
        raise argparse.ArgumentTypeError("point must be x,y")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--master-center", type=parse_point)
    parser.add_argument("--slave-center", type=parse_point)
    parser.add_argument("--master-radius", type=float, default=92.0)
    parser.add_argument("--slave-radius", type=float, default=92.0)
    parser.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    parser.add_argument("--paths", help="joint stereo path directory; replaces fixed centers")
    parser.add_argument("--start-offset-s", type=float, default=0.5)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--window-ms", type=float, default=40.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--maximum-rpm", type=float, default=1200.0)
    parser.add_argument("--marker-shell-mm", type=float, default=8.0)
    parser.add_argument("--marker-reprojection-px", type=float, default=3.5)
    parser.add_argument("--rotation-inlier", type=float, default=0.16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video")
    parser.add_argument("--marker-sets",
                        help="optional NPZ of per-frame 3-D candidates for pose research")
    args = parser.parse_args()

    calibration = StereoCalibration(args.calibration)
    master_reader = RawWindowReader(args.master_raw)
    slave_reader = RawWindowReader(args.slave_raw)
    step_us = int(args.step_ms * 1000)
    window_us = int(args.window_ms * 1000)
    if args.paths:
        path_dir = Path(args.paths)
        master_path = np.load(path_dir / "master_path.npz")
        slave_path = np.load(path_dir / "slave_path.npz")
        timestamps = master_path["timestamps_us"].astype(np.int64)
        centers_by_camera = (master_path["centers_px"].astype(np.float64),
                             slave_path["centers_px"].astype(np.float64))
        radii_by_camera = (master_path["radii_px"].astype(np.float64),
                           slave_path["radii_px"].astype(np.float64))
        points_m = (master_path["points_m"].astype(np.float64)
                    if "points_m" in master_path else np.array([
                        calibration.triangulate(a, b)[0]
                        for a, b in zip(*centers_by_camera)]))
        path_valid = (master_path["selected"] >= 0 if "selected" in master_path
                      else np.ones(len(timestamps), bool))
        if len(timestamps) > 1:
            step_us = int(np.median(np.diff(timestamps)))
        velocity_by_camera = tuple(
            np.column_stack([np.gradient(values[:, axis]) /
                             (np.gradient(timestamps) * 1e-6) for axis in range(2)])
            for values in centers_by_camera)
        center_error = float("nan")
    else:
        if args.master_center is None or args.slave_center is None:
            parser.error("fixed mode needs --master-center and --slave-center")
        common_start = max(master_reader.info.start_us, slave_reader.info.start_us)
        common_end = min(master_reader.info.end_us, slave_reader.info.end_us)
        start = common_start + int(args.start_offset_s * 1e6)
        end = min(common_end, start + int(args.duration_s * 1e6))
        timestamps = np.arange(start, end + 1, step_us, dtype=np.int64)
        centers_by_camera = tuple(np.repeat(value[None, :], len(timestamps), axis=0)
                                  for value in (args.master_center, args.slave_center))
        radii_by_camera = (np.full(len(timestamps), args.master_radius),
                           np.full(len(timestamps), args.slave_radius))
        velocity_by_camera = (np.zeros((len(timestamps), 2)), np.zeros((len(timestamps), 2)))
        center_m, center_error = calibration.triangulate(args.master_center, args.slave_center)
        points_m = np.repeat(center_m[None, :], len(timestamps), axis=0)
        path_valid = np.ones(len(timestamps), bool)
        if not np.isfinite(center_m).all():
            raise SystemExit("fixed ball centers cannot be triangulated")

    dot_config = {
        "normalized_patch_size": 128,
        "dot_detector": "blobs",
        "maximum_dots": 14,
        "dot_blob_threshold": 20,
        "dot_mask_radius_fraction": 0.72,
        "dot_blob_close_px": 3,
        "dot_blob_min_area_px": 5,
        "dot_blob_max_area_px": 350,
        "dot_blob_max_aspect_ratio": 5.0,
        "dot_min_distance_px": 12,
        "dot_canny_threshold": 25,
        "dot_vote_threshold": 5,
        "dot_min_radius_px": 2,
        "dot_max_radius_px": 8,
        "dot_minimum_ring_score": 0.10,
        "dot_nms_distance_px": 12,
    }
    detectors = (ImagePlaneSpinEstimator(dot_config, calibration.K1),
                 ImagePlaneSpinEstimator(dot_config, calibration.K2))
    readers = (master_reader, slave_reader)
    start, end = int(timestamps[0]), int(timestamps[-1])
    all_events = [reader.window(start - window_us, end) for reader in readers]
    event_times = [events["timestamp"].astype(np.int64) for events in all_events]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video = None
    if args.video:
        video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                1e3 / args.step_ms, (256, 128))
    fields = ["timestamp_us", "master_dots", "slave_dots", "markers_3d",
              "rotation_markers", "rotation_residual", "candidate_rpm_unvalidated",
              "axis_x", "axis_y", "axis_z"]
    rows = []
    marker_sets = []
    previous = None
    previous_timestamp = None
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_index, timestamp in enumerate(timestamps):
            point_sets = []
            views = []
            counts = []
            for camera, (events, times, detector) in enumerate(zip(
                    all_events, event_times, detectors)):
                center = centers_by_camera[camera][frame_index]
                radius = radii_by_camera[camera][frame_index]
                velocity = velocity_by_camera[camera][frame_index]
                lo = np.searchsorted(times, timestamp - window_us, side="right")
                hi = np.searchsorted(times, timestamp, side="right")
                patch = motion_compensated_ball_patch(
                    events[lo:hi], center, radius, 128, int(timestamp),
                    max(window_us / 2, 1), velocity, window_us)
                dots = (detector._dot_centers(detector._gray(patch))
                        if path_valid[frame_index] else np.empty((0, 3)))
                image_points = ((dots[:, :2] - 64.0) * (radius / 64.0) + center
                                if len(dots) else np.empty((0, 2)))
                point_sets.append(image_points)
                counts.append(len(dots))
                view = render_bgr(patch)
                for x, y, _ in dots:
                    cv2.circle(view, (round(x), round(y)), 3, (0, 255, 0), 1)
                views.append(view)
            markers = triangulate_sphere_markers(
                point_sets[0], point_sets[1], calibration, points_m[frame_index],
                radius_m=0.020, maximum_reprojection_px=args.marker_reprojection_px,
                maximum_shell_error_m=args.marker_shell_mm * 1e-3)
            marker_sets.append(markers)
            row = {name: "" for name in fields}
            row.update({"timestamp_us": int(timestamp), "master_dots": counts[0],
                        "slave_dots": counts[1], "markers_3d": len(markers.directions)})
            if previous is not None and previous_timestamp is not None:
                dt = (int(timestamp) - previous_timestamp) * 1e-6
                maximum_rotation = args.maximum_rpm * 2 * np.pi / 60 * dt
                rotation, old, new, residual = estimate_unordered_sphere_rotation(
                    previous, markers.directions,
                    maximum_rotation_rad=maximum_rotation,
                    inlier_chord_distance=args.rotation_inlier,
                    minimum_markers=3,
                    hypotheses=800,
                    seed=int(timestamp) & 0xFFFF)
                if len(old) >= 3:
                    rotation_vector = _rotation_vector(rotation)
                    omega = rotation_vector / dt
                    magnitude = float(np.linalg.norm(omega))
                    axis = omega / max(magnitude, 1e-12)
                    row.update({"rotation_markers": len(old), "rotation_residual": residual,
                                "candidate_rpm_unvalidated": magnitude * 60 / (2 * np.pi),
                                "axis_x": axis[0], "axis_y": axis[1], "axis_z": axis[2]})
            # Adjacent-frame estimation is deliberate: do not bridge a failed detection
            # with an incorrect dt while the marker identities remain unordered.
            previous = markers.directions
            previous_timestamp = int(timestamp)
            writer.writerow(row)
            rows.append(row)
            if video:
                video.write(np.hstack(views))
    if video:
        video.release()

    if args.marker_sets:
        marker_set_path = Path(args.marker_sets)
        marker_set_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.asarray([len(markers.directions) for markers in marker_sets], np.int32)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        def concatenate(name: str, shape: tuple[int, ...], dtype):
            values = [getattr(markers, name) for markers in marker_sets]
            return np.concatenate(values).astype(dtype, copy=False) if offsets[-1] else np.empty(shape, dtype)
        np.savez_compressed(
            marker_set_path,
            timestamps_us=timestamps,
            offsets=offsets,
            directions=concatenate("directions", (0, 3), np.float64),
            points_m=concatenate("points_m", (0, 3), np.float64),
            left_indices=concatenate("left_indices", (0,), np.int32),
            right_indices=concatenate("right_indices", (0,), np.int32),
            reprojection_error_px=concatenate("reprojection_error_px", (0,), np.float64),
            shell_error_m=concatenate("shell_error_m", (0,), np.float64),
        )

    valid = [row for row in rows if row["candidate_rpm_unvalidated"] != ""]
    rpms = np.asarray([row["candidate_rpm_unvalidated"] for row in valid], np.float64)
    axes = np.asarray([[row["axis_x"], row["axis_y"], row["axis_z"]] for row in valid],
                      np.float64)
    # Axes are signed by the estimated rotation, so use direct angular agreement.
    axis_resultant = float(np.linalg.norm(np.mean(axes, axis=0))) if len(axes) else 0.0
    report = {
        "purpose": "stereo_anonymous_marker_correspondence_diagnostic_not_validated_spin",
        "status": "spin_unvalidated",
        "spin_solver_allowed": False,
        "frames": len(rows),
        "frames_with_3d_markers": sum(int(row["markers_3d"]) >= 3 for row in rows),
        "rotation_states": len(valid),
        "center_m": (np.median(points_m[path_valid], axis=0).tolist()
                     if np.any(path_valid) else None),
        "center_reprojection_error_px": (float(center_error)
                                          if np.isfinite(center_error) else None),
        "spin_rpm": None,
        "candidate_rpm_unvalidated_median": float(np.median(rpms)) if len(rpms) else None,
        "candidate_rpm_unvalidated_p10": float(np.percentile(rpms, 10)) if len(rpms) else None,
        "candidate_rpm_unvalidated_p90": float(np.percentile(rpms, 90)) if len(rpms) else None,
        "axis_resultant": axis_resultant,
        "required_next_gate": "physical marker identities and controlled absolute-spin validation",
        "output": str(output.resolve()),
        "marker_sets": str(Path(args.marker_sets).resolve()) if args.marker_sets else None,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
