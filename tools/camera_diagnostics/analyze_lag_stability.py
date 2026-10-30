"""Estimate dual-camera lag repeatedly to distinguish fixed offset from clock drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_dual_camera_sync import estimate_lag
from record_dual_sync_evidence import read_raw_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("slave", type=Path)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--max-lag-us", type=int, default=2_000)
    parser.add_argument("--min-correlation", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    master = read_raw_counts(args.master, args.bin_us)
    slave = read_raw_counts(args.slave, args.bin_us)
    size = min(len(master["counts"]), len(slave["counts"]))
    m = np.log1p(master["counts"][:size].astype(float))
    s = np.log1p(slave["counts"][:size].astype(float))
    window = int(args.window_s * 1_000_000 / args.bin_us)
    step = int(args.step_s * 1_000_000 / args.bin_us)
    margin = int(1_000_000 / args.bin_us)
    rows = []
    for begin in range(margin, size - margin - window + 1, step):
        result = estimate_lag(m[begin:begin + window], s[begin:begin + window],
                              args.bin_us, args.max_lag_us)
        rows.append({
            "center_s": (begin + window / 2) * args.bin_us / 1_000_000,
            "lag_us": result.lag_us,
            "correlation": result.correlation,
        })
    accepted = [row for row in rows if row["correlation"] >= args.min_correlation]
    if accepted:
        times = np.array([row["center_s"] for row in accepted])
        lags = np.array([row["lag_us"] for row in accepted], dtype=float)
        median = float(np.median(lags))
        mad = float(np.median(np.abs(lags - median)))
        slope = float(np.polyfit(times, lags, 1)[0]) if len(lags) > 1 else float("nan")
    else:
        median = mad = slope = float("nan")
    report = {
        "bin_us": args.bin_us,
        "window_s": args.window_s,
        "minimum_correlation": args.min_correlation,
        "windows": rows,
        "accepted_windows": len(accepted),
        "median_lag_us": median,
        "lag_mad_us": mad,
        "drift_slope_us_per_s": slope,
        "suggested_slave_timestamp_correction_us": -median if np.isfinite(median) else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
