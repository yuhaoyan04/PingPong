"""Summarize stable multi-frame spin tracks from a pipeline CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_event_pipeline.quality import summarize_spin_segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--output")
    ap.add_argument("--minimum-samples", type=int, default=20)
    args = ap.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True, encoding="utf-8-sig",
                         filling_values=np.nan)
    names = set(data.dtype.names or ())
    if "master_omega_x" in names:
        master = np.column_stack((data["master_omega_x"], data["master_omega_y"], data["master_omega_z"]))
        slave = np.column_stack((data["slave_omega_master_x"], data["slave_omega_master_y"],
                                 data["slave_omega_master_z"]))
        master_valid = np.isfinite(master).all(axis=1)
        slave_valid = np.isfinite(slave).all(axis=1)
        timestamps = np.r_[data["timestamp_us"][master_valid], data["timestamp_us"][slave_valid]]
        omega = np.vstack((master[master_valid], slave[slave_valid]))
        quality = np.r_[data["master_spin_quality"][master_valid],
                        data["slave_spin_quality"][slave_valid]]
        order = np.argsort(timestamps, kind="stable")
        segments = summarize_spin_segments(
            timestamps[order], omega[order], quality[order], np.zeros(len(order)),
            minimum_samples=args.minimum_samples,
        )
        for segment in segments:
            lo, hi = segment["start_timestamp_us"], segment["end_timestamp_us"]
            camera_results = []
            for vectors, valid_mask, q in ((master, master_valid, data["master_spin_quality"]),
                                           (slave, slave_valid, data["slave_spin_quality"])):
                mask = valid_mask & (data["timestamp_us"] >= lo) & (data["timestamp_us"] <= hi)
                result = summarize_spin_segments(data["timestamp_us"][mask], vectors[mask], q[mask],
                                                 np.zeros(np.count_nonzero(mask)),
                                                 minimum_samples=max(4, args.minimum_samples // 4))
                camera_results.append(result[0] if result else None)
            segment["master_samples"] = 0 if camera_results[0] is None else camera_results[0]["samples"]
            segment["slave_samples"] = 0 if camera_results[1] is None else camera_results[1]["samples"]
            if all(camera_results):
                va = np.asarray(camera_results[0]["spin_axis_camera"]) * camera_results[0]["spin_rpm"]
                vb = np.asarray(camera_results[1]["spin_axis_camera"]) * camera_results[1]["spin_rpm"]
                denom = max(np.linalg.norm(0.5 * (va + vb)), 1.0)
                cross = float(np.linalg.norm(va - vb) / denom)
                segment["cross_camera_relative_vector_difference"] = cross
                segment["internally_consistent"] = bool(
                    segment["internally_consistent"] and cross <= 0.30)
            else:
                segment["cross_camera_relative_vector_difference"] = None
                segment["internally_consistent"] = False
    else:
        omega = np.column_stack((data["omega_x_rad_s"], data["omega_y_rad_s"], data["omega_z_rad_s"]))
        segments = summarize_spin_segments(
            data["timestamp_us"], omega, data["spin_quality"], data["spin_disagreement"],
            minimum_samples=args.minimum_samples,
        )
    for segment in segments:
        segment["physical_marker_identity_and_absolute_spin_validation"] = False
        segment["accepted"] = False
    report = {"source_csv": str(Path(args.csv).resolve()), "segments": segments,
              "accepted_segments": 0,
              "warning": ("Internal consistency is diagnostic only; all spin segments remain "
                          "unaccepted until physical-marker identity and controlled absolute-spin "
                          "validation pass.")}
    output = Path(args.output) if args.output else Path(args.csv).with_suffix(".summary.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
