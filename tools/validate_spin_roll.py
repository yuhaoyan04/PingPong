"""Validate visual spin against the no-slip rolling constraint v = omega * radius."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_event_pipeline.frames import TableFrame


def robust_segment_velocity(timestamp_us: np.ndarray, positions: np.ndarray,
                            minimum_pair_dt_s: float = 0.10) -> tuple[np.ndarray, float, float]:
    """Estimate one segment velocity from median pair slopes.

    Stereo depth noise makes adjacent-point derivatives unusable.  Pair slopes
    separated by at least 100 ms retain real translation while suppressing the
    amplification of disparity jitter.
    """
    t = (timestamp_us.astype(np.float64) - float(timestamp_us[0])) * 1e-6
    slopes = []
    for i in range(len(t) - 1):
        dt = t[i + 1:] - t[i]
        keep = dt >= minimum_pair_dt_s
        if np.any(keep):
            slopes.append((positions[i + 1:][keep] - positions[i]) / dt[keep, None])
    if not slopes:
        return np.full(3, np.nan), float("nan"), float("nan")
    velocity = np.median(np.vstack(slopes), axis=0)
    intercept = np.median(positions - t[:, None] * velocity, axis=0)
    residual = np.linalg.norm(positions - (intercept + t[:, None] * velocity), axis=1)
    rms = float(np.sqrt(np.median(residual ** 2)))
    duration = float(t[-1] - t[0])
    return velocity, rms, duration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ball-radius-m", type=float, default=0.020)
    ap.add_argument("--minimum-speed-mps", type=float, default=0.05)
    ap.add_argument("--minimum-quality", type=float, default=0.15)
    ap.add_argument("--table-frame", required=True,
                    help="球桌外参；严格滚动验收必须知道桌面法向")
    ap.add_argument("--output")
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True, encoding="utf-8-sig",
                         filling_values=np.nan)
    names = set(data.dtype.names or ())
    observed = data["spin_rpm"]
    quality = (data["spin_quality"] if "spin_quality" in names
               else np.ones(len(np.atleast_1d(data)), dtype=float))
    prefix = "measured_" if "measured_x_m" in names else ""
    positions = np.column_stack((data[prefix + "x_m"], data[prefix + "y_m"],
                                 data[prefix + "z_m"]))
    position_valid = np.isfinite(positions).all(axis=1) & np.isfinite(data["timestamp_us"])
    if np.count_nonzero(position_valid) >= 10:
        segment_velocity, trajectory_rms_m, segment_duration_s = robust_segment_velocity(
            data["timestamp_us"][position_valid], positions[position_valid])
    else:
        segment_velocity = np.full(3, np.nan)
        trajectory_rms_m = segment_duration_s = float("nan")
    table_frame = TableFrame(args.table_frame)
    segment_velocity_table = table_frame.vector(segment_velocity)
    horizontal_velocity = segment_velocity_table.copy(); horizontal_velocity[2] = 0.0
    segment_speed = float(np.linalg.norm(horizontal_velocity))
    reference_rpm = segment_speed * 60.0 / (2.0 * np.pi * args.ball_radius_m)
    valid = (np.isfinite(observed) & np.isfinite(quality) &
             (quality >= args.minimum_quality) & (segment_speed >= args.minimum_speed_mps))
    expected = np.full(np.count_nonzero(valid), reference_rpm)
    relative_error = np.abs(observed[valid] - expected) / np.maximum(expected, 1e-9)

    perpendicularity = np.array([], dtype=float)
    rolling_axis_alignment = np.array([], dtype=float)
    axis = np.column_stack((data["spin_axis_x"], data["spin_axis_y"], data["spin_axis_z"]))[valid]
    axis = (table_frame.R @ axis.T).T
    velocity = np.broadcast_to(segment_velocity_table, axis.shape)
    norms = np.linalg.norm(velocity, axis=1)
    finite = np.isfinite(axis).all(axis=1) & np.isfinite(velocity).all(axis=1) & (norms > 1e-6)
    if np.any(finite):
        perpendicularity = np.abs(np.sum(axis[finite] * velocity[finite], axis=1) / norms[finite])
        direction = horizontal_velocity / max(segment_speed, 1e-9)
        expected_axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
        axis_norm = np.linalg.norm(axis[finite], axis=1)
        rolling_axis_alignment = np.sum(axis[finite] * expected_axis, axis=1) / axis_norm

    count = int(np.count_nonzero(valid))
    median_error = float(np.median(relative_error)) if count else None
    p90_error = float(np.percentile(relative_error, 90)) if count else None
    median_perpendicularity = float(np.median(perpendicularity)) if len(perpendicularity) else None
    median_axis_alignment = (float(np.median(rolling_axis_alignment))
                             if len(rolling_axis_alignment) else None)
    passed = bool(count >= 20 and median_error is not None and median_error <= 0.20 and
                  p90_error is not None and p90_error <= 0.40 and
                  median_perpendicularity is not None and median_perpendicularity <= 0.30 and
                  median_axis_alignment is not None and median_axis_alignment >= 0.85 and
                  trajectory_rms_m <= 0.02 and segment_speed * segment_duration_s >= 0.08)
    report = {
        "status": "absolute_spin_valid" if passed else "needs_review",
        "passed": passed,
        "accepted_samples": count,
        "ball_radius_m": args.ball_radius_m,
        "observed_rpm_median": float(np.median(observed[valid])) if count else None,
        "rolling_reference_rpm_median": reference_rpm if np.isfinite(reference_rpm) else None,
        "segment_velocity_master_mps": segment_velocity.tolist(),
        "segment_velocity_table_mps": segment_velocity_table.tolist(),
        "segment_speed_mps": segment_speed if np.isfinite(segment_speed) else None,
        "segment_duration_s": segment_duration_s if np.isfinite(segment_duration_s) else None,
        "fitted_travel_m": (segment_speed * segment_duration_s
                            if np.isfinite(segment_speed * segment_duration_s) else None),
        "trajectory_robust_rms_m": trajectory_rms_m if np.isfinite(trajectory_rms_m) else None,
        "relative_error_median": median_error,
        "relative_error_p90": p90_error,
        "axis_velocity_abs_dot_median": median_perpendicularity,
        "rolling_axis_signed_alignment_median": median_axis_alignment,
        "criteria": {"minimum_samples": 20, "median_relative_error_max": 0.20,
                     "p90_relative_error_max": 0.40,
                     "axis_velocity_abs_dot_median_max": 0.30,
                     "rolling_axis_signed_alignment_median_min": 0.85,
                     "trajectory_robust_rms_m_max": 0.02,
                     "minimum_fitted_travel_m": 0.08},
    }
    output = Path(args.output) if args.output else Path(args.csv).with_suffix(".spin_roll.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
