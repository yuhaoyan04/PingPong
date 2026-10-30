"""Second-pass stereo trajectory and spin solve using smoothed image paths."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.config import load_config
from pingpong_event_pipeline.events import RawWindowReader, motion_compensated_ball_patch, render_bgr
from pingpong_event_pipeline.models import Track2D
from pingpong_event_pipeline.spin import ImagePlaneSpinEstimator, fuse_optical_axis_spin
from pingpong_event_pipeline.stereo import StereoCalibration


FIELDS = ["timestamp_us", "master_x", "master_y", "master_radius", "slave_x", "slave_y",
          "slave_radius", "measured_x_m", "measured_y_m", "measured_z_m",
          "reprojection_error_px", "master_event_count", "slave_event_count",
          "master_spin_rpm", "slave_spin_rpm", "master_spin_features", "slave_spin_features",
          "master_spin_quality", "slave_spin_quality", "omega_x_rad_s", "omega_y_rad_s",
          "omega_z_rad_s", "spin_rpm", "spin_axis_x", "spin_axis_y", "spin_axis_z",
          "spin_quality", "spin_disagreement",
          "master_omega_x", "master_omega_y", "master_omega_z",
          "slave_omega_x", "slave_omega_y", "slave_omega_z"]


def load_path(path: Path):
    data = np.load(path)
    timestamps = data["timestamps_us"].astype(np.int64)
    centers = data["centers_px"].astype(np.float64)
    radii = data["radii_px"].astype(np.float64)
    dt = np.gradient(timestamps).astype(np.float64) * 1e-6
    velocity = np.column_stack((np.gradient(centers[:, 0]) / dt,
                                np.gradient(centers[:, 1]) / dt))
    return timestamps, centers, radii, velocity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-raw", required=True)
    ap.add_argument("--slave-raw", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--config", default="config/offline_fast_ball.yaml")
    ap.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--patch-video")
    ap.add_argument("--minimum-events", type=int, default=120)
    args = ap.parse_args()
    cfg = load_config(args.config); spin_cfg = cfg["spin"]
    calibration = StereoCalibration(args.calibration)
    paths = Path(args.paths)
    mt, mc, mr, mv = load_path(paths / "master_path.npz")
    st, sc, sr, sv = load_path(paths / "slave_path.npz")
    if not np.array_equal(mt, st):
        raise SystemExit("master/slave path timestamps differ")
    master_reader, slave_reader = RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)
    age = int(spin_cfg["window_us"]); start, end = int(mt[0] - age), int(mt[-1])
    master_events = master_reader.window(start, end); slave_events = slave_reader.window(start, end)
    master_times = master_events["timestamp"].astype(np.int64)
    slave_times = slave_events["timestamp"].astype(np.int64)
    master_estimator = ImagePlaneSpinEstimator(spin_cfg, calibration.K1)
    slave_estimator = ImagePlaneSpinEstimator(spin_cfg, calibration.K2)
    size = int(spin_cfg["normalized_patch_size"])
    writer_video = None
    if args.patch_video:
        writer_video = cv2.VideoWriter(args.patch_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                       1e6 / np.median(np.diff(mt)), (size * 2, size))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    rows = []; stereo_count = fused_count = 0
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for i, timestamp in enumerate(mt):
            row = {name: "" for name in FIELDS}; row["timestamp_us"] = int(timestamp)
            patches = []; estimates = []
            for prefix, events, event_times, center, radius, velocity, estimator, width, height in (
                ("master", master_events, master_times, mc[i], mr[i], mv[i], master_estimator,
                 master_reader.info.width, master_reader.info.height),
                ("slave", slave_events, slave_times, sc[i], sr[i], sv[i], slave_estimator,
                 slave_reader.info.width, slave_reader.info.height)):
                lo = np.searchsorted(event_times, timestamp - age, side="right")
                hi = np.searchsorted(event_times, timestamp, side="right")
                current_events = events[lo:hi]
                visible = (-radius < center[0] < width + radius and
                           -radius < center[1] < height + radius and radius >= spin_cfg["minimum_radius_px"])
                patch = motion_compensated_ball_patch(
                    current_events, center, radius, size, int(timestamp), spin_cfg["sae_tau_us"],
                    velocity, age) if visible else np.zeros((2, size, size), np.float32)
                track = (Track2D(int(timestamp), center[0], center[1], velocity[0], velocity[1],
                                 radius, 1.0, True)
                         if visible and len(current_events) >= args.minimum_events else None)
                estimate = estimator.update(patch, track, normalized_patch=True)
                patches.append(patch); estimates.append(estimate)
                row.update({f"{prefix}_x": center[0], f"{prefix}_y": center[1],
                            f"{prefix}_radius": radius, f"{prefix}_event_count": len(current_events)})
                if estimate:
                    row.update({f"{prefix}_spin_rpm": estimate.rpm,
                                f"{prefix}_spin_features": estimate.feature_count,
                                f"{prefix}_spin_quality": estimate.quality})
                    for axis, value in zip("xyz", estimate.omega_camera_rad_s):
                        row[f"{prefix}_omega_{axis}"] = value
            point, reprojection = calibration.triangulate(mc[i], sc[i])
            if (np.isfinite(point).all() and cfg["stereo"]["minimum_depth_m"] <= point[2] <=
                    cfg["stereo"]["maximum_depth_m"] and
                    reprojection <= cfg["stereo"]["max_reprojection_error_px"]):
                row.update({"measured_x_m": point[0], "measured_y_m": point[1],
                            "measured_z_m": point[2], "reprojection_error_px": reprojection})
                stereo_count += 1
            if estimates[0] is not None and estimates[1] is not None:
                right_in_left = calibration.R.T @ estimates[1].omega_camera_rad_s
                omega, quality = fuse_optical_axis_spin(
                    estimates[0].omega_camera_rad_s, estimates[1].omega_camera_rad_s,
                    calibration.R.T, estimates[0].quality, estimates[1].quality)
                disagreement = float(np.linalg.norm(estimates[0].omega_camera_rad_s - right_in_left) /
                                     max(np.linalg.norm(omega), 1.0))
                row["spin_disagreement"] = disagreement
                if (quality >= spin_cfg["minimum_fused_quality"] and
                        disagreement <= spin_cfg["maximum_relative_disagreement"]):
                    magnitude = float(np.linalg.norm(omega)); axis = omega / max(magnitude, 1e-12)
                    row.update({"omega_x_rad_s": omega[0], "omega_y_rad_s": omega[1],
                                "omega_z_rad_s": omega[2], "spin_rpm": magnitude * 60 / (2*np.pi),
                                "spin_axis_x": axis[0], "spin_axis_y": axis[1], "spin_axis_z": axis[2],
                                "spin_quality": quality})
                    fused_count += 1
            writer.writerow(row); rows.append(row)
            if writer_video:
                view = np.hstack([render_bgr(patch) for patch in patches])
                writer_video.write(view)
    if writer_video:
        writer_video.release()
    report = {"rows": len(rows), "stereo_states": stereo_count, "fused_spin_states": fused_count,
              "output": str(output.resolve()),
              "patch_video": str(Path(args.patch_video).resolve()) if args.patch_video else None}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
