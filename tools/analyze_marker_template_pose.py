"""Test fixed-template pose observability from per-frame 3-D sphere candidates."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.stereo_spin import (
    estimate_sphere_template_pose, SphereTemplateMatcher)


def marker_frame(data, index: int) -> np.ndarray:
    first, last = data["offsets"][index:index + 2]
    return data["directions"][int(first):int(last)]


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    vector, _ = cv2.Rodrigues(second @ first.T)
    return float(np.linalg.norm(vector))


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="固定匿名球面模板逐窗姿态实验；不输出转速")
    parser.add_argument("marker_sets")
    parser.add_argument("--reference-index", type=int)
    parser.add_argument("--inlier-chord", type=float, default=.08)
    parser.add_argument("--minimum-markers", type=int, default=3)
    parser.add_argument("--hypotheses", type=int, default=5000)
    parser.add_argument("--method", choices=("ransac", "hash"), default="hash")
    parser.add_argument("--signature-tolerance", type=float, default=.035)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.marker_sets)
    timestamps = data["timestamps_us"].astype(np.int64)
    counts = np.diff(data["offsets"]).astype(np.int32)
    if args.reference_index is None:
        eligible = np.flatnonzero(counts >= max(5, args.minimum_markers))
        if not len(eligible):
            raise SystemExit("没有包含至少 5 个三维候选的参考帧")
        quality = []
        for index in eligible:
            first, last = data["offsets"][index:index + 2]
            reprojection = data["reprojection_error_px"][int(first):int(last)]
            shell = data["shell_error_m"][int(first):int(last)]
            quality.append(counts[index] - .2 * np.median(reprojection)
                           - 20 * np.median(shell))
        reference_index = int(eligible[int(np.argmax(quality))])
    else:
        reference_index = int(args.reference_index)
    template = marker_frame(data, reference_index)
    matcher = SphereTemplateMatcher(
        template, signature_tolerance=args.signature_tolerance,
        inlier_chord_distance=args.inlier_chord,
        minimum_markers=args.minimum_markers,
        maximum_hypotheses=args.hypotheses)

    rows = []
    rotations: dict[int, np.ndarray] = {}
    hypotheses_tested = 0
    for index, timestamp in enumerate(timestamps):
        observations = marker_frame(data, index)
        if args.method == "hash":
            rotation, template_indices, observation_indices, residual, tested = (
                matcher.match(observations))
            hypotheses_tested += tested
        else:
            rotation, template_indices, observation_indices, residual = (
                estimate_sphere_template_pose(
                    template, observations,
                    inlier_chord_distance=args.inlier_chord,
                    minimum_markers=args.minimum_markers,
                    hypotheses=args.hypotheses,
                    seed=(int(timestamp) ^ 0x5A17) & 0xFFFFFFFF))
            hypotheses_tested += args.hypotheses
        valid = len(template_indices) >= args.minimum_markers
        if valid:
            rotations[index] = rotation
            vector, _ = cv2.Rodrigues(rotation)
            vector = vector.reshape(3)
        else:
            vector = np.full(3, np.nan)
        rows.append({
            "frame_index": index,
            "timestamp_us": int(timestamp),
            "observed_markers": len(observations),
            "template_inliers": len(template_indices),
            "pose_residual_chord": residual if valid else "",
            "candidate_pose_rotvec_x": vector[0] if valid else "",
            "candidate_pose_rotvec_y": vector[1] if valid else "",
            "candidate_pose_rotvec_z": vector[2] if valid else "",
            "adjacent_pose_jump_rad": "",
        })
    adjacent_jumps = []
    for index in range(1, len(rows)):
        if index in rotations and index - 1 in rotations:
            jump = rotation_distance(rotations[index - 1], rotations[index])
            rows[index]["adjacent_pose_jump_rad"] = jump
            adjacent_jumps.append(jump)

    valid_rows = [row for row in rows
                  if row["template_inliers"] >= args.minimum_markers]
    residuals = np.asarray(
        [row["pose_residual_chord"] for row in valid_rows], np.float64)
    jumps = np.asarray(adjacent_jumps, np.float64)
    valid_fraction = len(valid_rows) / max(1, len(rows))
    consistency = bool(
        valid_fraction >= .8 and len(jumps) >= 100
        and np.percentile(jumps, 90) <= .1
        and np.median(residuals) <= args.inlier_chord * .6)
    report = {
        "purpose": "anonymous_bootstrap_template_pose_observability_not_spin",
        "status": "template_pose_unvalidated",
        "spin_solver_allowed": False,
        "marker_sets": str(Path(args.marker_sets).resolve()),
        "frames": len(rows),
        "reference_index": reference_index,
        "reference_markers": len(template),
        "method": args.method,
        "hypotheses_tested": hypotheses_tested,
        "pose_frames": len(valid_rows),
        "pose_fraction": valid_fraction,
        "pose_residual_chord_median": (float(np.median(residuals))
                                       if len(residuals) else None),
        "adjacent_pose_pairs": len(jumps),
        "adjacent_pose_jump_rad_median": (float(np.median(jumps))
                                           if len(jumps) else None),
        "adjacent_pose_jump_rad_p90": (float(np.percentile(jumps, 90))
                                        if len(jumps) else None),
        "large_jump_fraction_over_0p1_rad": (float(np.mean(jumps > .1))
                                              if len(jumps) else None),
        "internal_template_consistency_pass": consistency,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "reference candidates are anonymous and may include false stereo pairs",
            "the reference frame covers only its visible hemisphere",
            "a consistent anonymous template is not a calibrated physical ball template",
            "no angular velocity or RPM is produced",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
