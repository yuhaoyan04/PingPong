"""Audit stereo ambiguity of anonymous marker proposals using calibration only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from annotate_dual_marker_identities import patch_to_full
from pingpong_event_pipeline.stereo import StereoCalibration
from visualize_dual_annotation_pack import load_proposals, load_records


def summarize(values):
    data = np.asarray(values, np.float64)
    if not len(data):
        return {"median": None, "p90": None, "maximum": None}
    return {"median": float(np.median(data)), "p90": float(np.percentile(data, 90)),
            "maximum": float(np.max(data))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用极线约束审计双路匿名标记候选；不建立物理 ID、不计算转速")
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--master-proposals", type=Path, required=True)
    parser.add_argument("--slave-proposals", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration/stereo_calibration.npz"))
    parser.add_argument("--maximum-error-px", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    master_records = load_records(args.master_annotations)
    slave_records = load_records(args.slave_annotations)
    master_proposals = load_proposals(args.master_proposals)
    slave_proposals = load_proposals(args.slave_proposals)
    calibration = StereoCalibration(args.calibration)
    image_size = int(json.loads(args.master_annotations.read_text(
        encoding="utf-8"))["image_size_px"][0])
    timestamps = sorted(set(master_records) & set(slave_records))
    rows = []
    assigned_errors = []
    assigned_counts = []
    ambiguous_master = ambiguous_slave = total_master = total_slave = 0
    for timestamp in timestamps:
        mp = master_proposals.get(timestamp, [])
        sp = slave_proposals.get(timestamp, [])
        left = np.asarray([patch_to_full([item["x"], item["y"]],
                                        master_records[timestamp], image_size) for item in mp])
        right = np.asarray([patch_to_full([item["x"], item["y"]],
                                         slave_records[timestamp], image_size) for item in sp])
        costs = np.empty((len(left), len(right)), np.float64)
        for i, left_point in enumerate(left):
            for j, right_point in enumerate(right):
                costs[i, j] = calibration.symmetric_epipolar_error(left_point, right_point)
        pairs = []
        if costs.size:
            master_indices, slave_indices = linear_sum_assignment(costs)
            for i, j in zip(master_indices, slave_indices):
                error = float(costs[i, j])
                if error <= args.maximum_error_px:
                    pairs.append({"master_index": int(i), "slave_index": int(j),
                                  "symmetric_epipolar_error_px": error})
                    assigned_errors.append(error)
            total_master += len(left); total_slave += len(right)
            ambiguous_master += int(np.sum(np.sum(costs <= args.maximum_error_px, axis=1) > 1))
            ambiguous_slave += int(np.sum(np.sum(costs <= args.maximum_error_px, axis=0) > 1))
        assigned_counts.append(len(pairs))
        rows.append({
            "timestamp_us": timestamp,
            "master_candidates": len(mp),
            "slave_candidates": len(sp),
            "assigned_pairs_within_gate": len(pairs),
            "pairs": pairs,
        })
    report = {
        "purpose": "anonymous_stereo_marker_epipolar_ambiguity_audit_no_identity_no_spin",
        "frames": len(timestamps),
        "maximum_error_px": args.maximum_error_px,
        "assigned_pairs_per_frame": summarize(assigned_counts),
        "assigned_epipolar_error_px": summarize(assigned_errors),
        "master_candidates_with_multiple_admissible_slave_matches_fraction": (
            ambiguous_master / max(1, total_master)),
        "slave_candidates_with_multiple_admissible_master_matches_fraction": (
            ambiguous_slave / max(1, total_slave)),
        "identity_proven": False,
        "spin_solver_allowed": False,
        "interpretation": (
            "Epipolar geometry narrows manual correspondence but cannot establish physical identity; "
            "shared human IDs or a calibrated marker template remain required."),
        "frames_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames_detail"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
