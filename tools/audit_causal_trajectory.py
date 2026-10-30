"""Compare the causal tracker with the high-quality fitted stereo path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--stereo-path", required=True)
    parser.add_argument("--first-index", type=int, required=True)
    parser.add_argument("--last-index", type=int, required=True)
    parser.add_argument("--processing-wall-s", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = np.load(args.stereo_path)
    timestamps = path["timestamps_us"].astype(np.int64)
    first_t, last_t = timestamps[args.first_index], timestamps[args.last_index]
    frame = pd.read_csv(args.csv)
    measured = frame[(frame["measured_x_m"].notna()) &
                     (frame["timestamp_us"] >= first_t) & (frame["timestamp_us"] <= last_t)]
    points = path["points_m"][args.first_index:args.last_index + 1]
    reference_t = timestamps[args.first_index:args.last_index + 1]
    dt = np.gradient(reference_t).astype(np.float64) * 1e-6
    velocity = np.column_stack([np.gradient(points[:, i]) / dt for i in range(3)])
    reference_speed = float(np.median(np.linalg.norm(velocity, axis=1)))
    causal_speeds = measured.loc[measured["speed_mps"] > 0, "speed_mps"].to_numpy()
    causal_speed = float(np.median(causal_speeds)) if len(causal_speeds) else 0.0
    relative_error = abs(causal_speed - reference_speed) / max(reference_speed, 1e-9)
    expected = max(1, round((last_t - first_t) /
                            np.median(np.diff(frame["timestamp_us"].to_numpy()))) + 1)
    reprojection_p90 = float(measured["reprojection_error_px"].quantile(.9))
    gates = {
        "at_least_30_stereo_points": len(measured) >= 30,
        "coverage_at_least_70pct": len(measured) / expected >= .70,
        "median_speed_within_15pct_of_quality_fit": relative_error <= .15,
        "reprojection_p90_le_3px": reprojection_p90 <= 3.0,
    }
    report = {
        "status": "trajectory_valid" if all(gates.values()) else "needs_review",
        "stereo_points": len(measured), "expected_windows": expected,
        "coverage": len(measured) / expected,
        "causal_speed_mps_median": causal_speed,
        "quality_fit_speed_mps_median": reference_speed,
        "relative_speed_difference": relative_error,
        "reprojection_error_p90_px": reprojection_p90,
        "gates": gates,
    }
    if args.processing_wall_s is not None:
        data_s = (int(frame["timestamp_us"].iloc[-1]) - int(frame["timestamp_us"].iloc[0])) * 1e-6
        report.update({"processed_data_s": data_s, "processing_wall_s": args.processing_wall_s,
                       "realtime_factor": data_s / args.processing_wall_s})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
