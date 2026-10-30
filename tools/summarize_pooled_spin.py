"""Create an auditable multi-baseline trajectory/spin measurement report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.frames import TableFrame, spin_components_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spin-json", nargs="+", required=True)
    parser.add_argument("--stereo-path", required=True)
    parser.add_argument("--first-index", type=int, required=True)
    parser.add_argument("--last-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states-csv")
    parser.add_argument("--path-video")
    parser.add_argument("--preview")
    parser.add_argument("--table-frame", help="主相机到球桌坐标变换；用于上旋/下旋/侧旋分解")
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.spin_json]
    rpms = np.asarray([run["rpm"] for run in runs], np.float64)
    axes = np.asarray([run["axis_master"] for run in runs], np.float64)
    reference = axes[0]
    axes[np.sum(axes * reference, axis=1) < 0] *= -1
    mean_axis = np.mean(axes, axis=0); mean_axis /= np.linalg.norm(mean_axis)
    axis_deviation = np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1)))
    path = np.load(args.stereo_path)
    timestamps = path["timestamps_us"].astype(np.int64)
    points = path["points_m"].astype(np.float64)
    sl = slice(args.first_index, args.last_index + 1)
    dt = np.gradient(timestamps[sl]).astype(np.float64) * 1e-6
    velocity = np.column_stack([np.gradient(points[sl, axis]) / dt for axis in range(3)])
    speed = np.linalg.norm(velocity, axis=1)
    rpm_cv = float(np.std(rpms) / max(np.mean(rpms), 1e-9))
    median_camera_axis = float(np.median([run.get("camera_axis_angle_deg", np.inf) for run in runs]))
    median_camera_rpm = float(np.median([
        run.get("camera_relative_rpm_difference", np.inf) for run in runs]))
    minimum_inliers = int(min(run["inliers"] for run in runs))
    nominal_step_us = float(np.median(np.diff(timestamps)))
    detection_step_us = float(path["detection_step_us"]) if "detection_step_us" in path else nominal_step_us
    internal_gates = {
        "full_rate_path_sampling": detection_step_us <= nominal_step_us + 1e-6,
        "at_least_three_time_baselines": len(runs) >= 3,
        "multi_baseline_rpm_cv_le_10pct": rpm_cv <= .10,
        "multi_baseline_axis_max_deviation_le_15deg": float(np.max(axis_deviation)) <= 15,
        "median_camera_axis_difference_le_30deg": median_camera_axis <= 30,
        "median_camera_rpm_difference_le_25pct": median_camera_rpm <= .25,
        "minimum_500_inliers_per_baseline": minimum_inliers >= 500,
        "well_conditioned_geometry": max(run["condition"] for run in runs) <= 4,
    }
    internally_consistent = all(internal_gates.values())
    gates = {
        **internal_gates,
        # This fail-closed gate is intentionally not configurable.  The current
        # LK/pooled solver has not passed physical-dot identity and controlled
        # absolute-spin validation after the ball-disc background fix.
        "physical_marker_identity_and_absolute_spin_validation": False,
    }
    candidate_rpm = float(np.median(rpms))
    report = {
        "status": "spin_unvalidated" if internally_consistent else "needs_review",
        "spin_rpm": None,
        "candidate_spin_rpm_unvalidated": candidate_rpm,
        "spin_rpm_method_p10": float(np.percentile(rpms, 10)),
        "spin_rpm_method_p90": float(np.percentile(rpms, 90)),
        "spin_rpm_cv": rpm_cv,
        "spin_axis_master_camera": mean_axis.tolist(),
        "axis_max_method_deviation_deg": float(np.max(axis_deviation)),
        "median_left_right_axis_difference_deg": median_camera_axis,
        "median_left_right_rpm_difference": median_camera_rpm,
        "speed_mps_median": float(np.median(speed)),
        "speed_mps_p10": float(np.percentile(speed, 10)),
        "speed_mps_p90": float(np.percentile(speed, 90)),
        "duration_ms": (int(timestamps[args.last_index]) -
                        int(timestamps[args.first_index])) * 1e-3,
        "minimum_inliers": minimum_inliers,
        "path_detection_step_us": detection_step_us,
        "gates": gates,
        "internal_consistency_only": internally_consistent,
        "source_runs": [str(Path(path).resolve()) for path in args.spin_json],
        "limitations": [
            "Candidate RPM is diagnostic only and is never promoted to spin_rpm until physical-marker identity and controlled absolute-spin validation pass.",
            "Internal multi-camera/multi-baseline consistency does not prove physical-dot correspondence or absolute RPM accuracy.",
        ],
    }
    table_frame = TableFrame(args.table_frame) if args.table_frame else None
    table_spin = None
    if table_frame is not None and report["spin_rpm"] is not None:
        omega_master = mean_axis * report["spin_rpm"] * (2.0 * np.pi / 60.0)
        omega_table = table_frame.vector(omega_master)
        velocity_table = table_frame.vector(np.median(velocity, axis=0))
        table_spin = spin_components_table(omega_table, velocity_table)
        report["table_coordinates"] = {
            "spin_axis": (omega_table / np.linalg.norm(omega_table)).tolist(),
            "mean_velocity_mps": velocity_table.tolist(),
            **table_spin,
        }
        report["limitations"].append(
            "Spin labels inherit the physical +X/+Y directions chosen during table-frame calibration.")
    else:
        report["limitations"][:0] = [
            "Axis is expressed in the master-camera coordinate system.",
            "Topspin/backspin labels require a camera-to-table extrinsic calibration.",
        ]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.states_csv:
        fields = ["timestamp_us", "x_m", "y_m", "z_m", "vx_mps", "vy_mps",
                  "vz_mps", "speed_mps", "spin_rpm", "spin_axis_x",
                  "spin_axis_y", "spin_axis_z", "status"]
        if table_spin is not None:
            fields += ["table_x_m", "table_y_m", "table_z_m", "table_vx_mps",
                       "table_vy_mps", "table_vz_mps", "table_spin_axis_x",
                       "table_spin_axis_y", "table_spin_axis_z", "topspin_signed_rpm",
                       "vertical_sidespin_signed_rpm", "gyrospin_signed_rpm", "dominant_spin"]
        states_path = Path(args.states_csv); states_path.parent.mkdir(parents=True, exist_ok=True)
        with states_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for offset, index in enumerate(range(args.first_index, args.last_index + 1)):
                row = {
                    "timestamp_us": int(timestamps[index]), "x_m": points[index, 0],
                    "y_m": points[index, 1], "z_m": points[index, 2],
                    "vx_mps": velocity[offset, 0], "vy_mps": velocity[offset, 1],
                    "vz_mps": velocity[offset, 2], "speed_mps": speed[offset],
                    "spin_rpm": report["spin_rpm"], "spin_axis_x": None,
                    "spin_axis_y": None, "spin_axis_z": None,
                    "status": report["status"]}
                if table_spin is not None:
                    table_point = table_frame.point(points[index])
                    table_velocity = table_frame.vector(velocity[offset])
                    table_axis = table_frame.vector(mean_axis)
                    row.update({"table_x_m": table_point[0], "table_y_m": table_point[1],
                                "table_z_m": table_point[2], "table_vx_mps": table_velocity[0],
                                "table_vy_mps": table_velocity[1], "table_vz_mps": table_velocity[2],
                                "table_spin_axis_x": table_axis[0],
                                "table_spin_axis_y": table_axis[1],
                                "table_spin_axis_z": table_axis[2],
                                "topspin_signed_rpm": table_spin["topspin_signed_rpm"],
                                "vertical_sidespin_signed_rpm": table_spin["vertical_sidespin_signed_rpm"],
                                "gyrospin_signed_rpm": table_spin["gyrospin_signed_rpm"],
                                "dominant_spin": table_spin["dominant_spin"]})
                writer.writerow(row)
        report["states_csv"] = str(states_path.resolve())
    if args.path_video and args.preview:
        capture = cv2.VideoCapture(args.path_video)
        fps = capture.get(cv2.CAP_PROP_FPS) or 1000.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        preview_path = Path(args.preview); preview_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.first_index <= index <= args.last_index:
                offset = index - args.first_index
                cv2.putText(frame, f"speed {speed[offset]:.2f} m/s", (24, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 255, 60), 2, cv2.LINE_AA)
                cv2.putText(frame, "spin UNVALIDATED", (24, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "physical marker/absolute-spin gate failed", (24, 138),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 100, 255), 2, cv2.LINE_AA)
                writer.write(frame)
            index += 1
        capture.release(); writer.release()
        report["preview"] = str(preview_path.resolve())
    # Rewrite after optional artifacts were added.
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
