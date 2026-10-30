"""Solve 3-D spin from a validated physical template and stereo marker sets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.stereo_spin import (
    SphereTemplatePoseTracker, estimate_spin_from_template_poses)


def time_ranges_overlap(first_start, first_end, second_start, second_end) -> bool:
    if any(value is None for value in (first_start, first_end, second_start, second_end)):
        return False
    return max(int(first_start), int(second_start)) <= min(int(first_end), int(second_end))


def marker_frame(data, index: int) -> np.ndarray:
    begin, end = int(data["offsets"][index]), int(data["offsets"][index + 1])
    return data["directions"][begin:end].astype(np.float64)


def rejected_report(reason: str, template_report: Path, marker_sets: Path) -> dict:
    return {
        "purpose": "validated_physical_template_3d_spin_measurement",
        "status": "rejected",
        "spin_solver_allowed": False,
        "spin_rpm": None,
        "omega_rad_s_camera": None,
        "axis_camera": None,
        "reason": reason,
        "template_report": str(template_report.resolve()),
        "marker_sets": str(marker_sets.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用已验收真实球面模板跟踪绝对姿态并解算三维转速")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-report", type=Path, required=True)
    parser.add_argument("--marker-sets", type=Path, required=True)
    parser.add_argument("--minimum-markers", type=int, default=4)
    parser.add_argument("--minimum-pose-fraction", type=float, default=.60)
    parser.add_argument("--signature-tolerance", type=float, default=.035)
    parser.add_argument("--inlier-chord", type=float, default=.06)
    parser.add_argument("--allow-calibration-overlap", action="store_true",
                        help="diagnostic only; formal measurement rejects calibration overlap")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.template_report.read_text(encoding="utf-8"))
    observations = np.load(args.marker_sets)
    timestamps = observations["timestamps_us"].astype(np.int64)
    reasons = []
    if not metadata.get("physical_template_geometry_accepted", False):
        reasons.append("physical template geometry gate did not pass")
    if not metadata.get("candidate_tracking_identity_accepted", False):
        reasons.append("candidate tracking identity gate did not pass")
    if not metadata.get("spin_solver_allowed", False):
        reasons.append("template report does not allow the spin solver")
    overlap = time_ranges_overlap(
        metadata.get("calibration_timestamp_start_us"),
        metadata.get("calibration_timestamp_end_us"),
        int(timestamps[0]) if len(timestamps) else None,
        int(timestamps[-1]) if len(timestamps) else None)
    if overlap and not args.allow_calibration_overlap:
        reasons.append("measurement timestamps overlap the template calibration sequence")
    template_data = np.load(args.template)
    template = template_data["directions"].astype(np.float64)
    if len(template) < args.minimum_markers:
        reasons.append(f"template has fewer than {args.minimum_markers} markers")
    if reasons:
        report = rejected_report("; ".join(reasons), args.template_report, args.marker_sets)
        report["calibration_measurement_overlap"] = overlap
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    tracker = SphereTemplatePoseTracker(
        template, signature_tolerance=args.signature_tolerance,
        inlier_chord_distance=args.inlier_chord,
        minimum_markers=args.minimum_markers,
        maximum_hypotheses=5000, early_stop_unmatched_observations=2)
    pose_timestamps = []
    poses = []
    rows = []
    global_frames = local_frames = 0
    for index, timestamp in enumerate(timestamps):
        current = marker_frame(observations, index)
        pose, template_indices, _, residual, mode, tested = tracker.update(
            int(timestamp), current)
        valid = len(template_indices) >= args.minimum_markers and residual <= args.inlier_chord
        if valid:
            pose_timestamps.append(int(timestamp)); poses.append(pose)
            vector, _ = cv2.Rodrigues(pose)
            global_frames += mode == "global"; local_frames += mode == "local"
        else:
            vector = np.full((3, 1), np.nan)
        rows.append({
            "timestamp_us": int(timestamp),
            "observed_markers": len(current),
            "pose_valid": valid,
            "template_inliers": len(template_indices),
            "pose_residual_chord": residual if np.isfinite(residual) else "",
            "pose_mode": mode,
            "hypotheses_tested": tested,
            "pose_rotvec_x": vector[0, 0] if valid else "",
            "pose_rotvec_y": vector[1, 0] if valid else "",
            "pose_rotvec_z": vector[2, 0] if valid else "",
        })
    pose_fraction = len(poses) / max(1, len(timestamps))
    spin = estimate_spin_from_template_poses(
        np.asarray(pose_timestamps, np.int64), np.asarray(poses, np.float64),
        minimum_pairs=6, minimum_span_s=.006,
        inlier_threshold_rad=.04, maximum_rms_rad=.025,
        minimum_axis_resultant=.85)
    accepted = bool(pose_fraction >= args.minimum_pose_fraction and spin.accepted)
    report = {
        "purpose": "validated_physical_template_3d_spin_measurement",
        "status": "accepted" if accepted else "rejected",
        "spin_solver_allowed": accepted,
        "spin_rpm": spin.rpm if accepted else None,
        "omega_rad_s_camera": spin.omega_rad_s.tolist() if accepted else None,
        "axis_camera": spin.axis.tolist() if accepted else None,
        "reason": (spin.reason if pose_fraction >= args.minimum_pose_fraction else
                   f"pose fraction {pose_fraction:.3f} is below {args.minimum_pose_fraction:.3f}"),
        "input_frames": len(timestamps),
        "valid_pose_frames": len(poses),
        "pose_fraction": pose_fraction,
        "global_match_frames": global_frames,
        "local_track_frames": local_frames,
        "spin_inlier_pairs": spin.inlier_pairs,
        "spin_residual_rms_rad": spin.residual_rms_rad,
        "axis_resultant": spin.axis_resultant,
        "calibration_measurement_overlap": overlap,
        "template_report": str(args.template_report.resolve()),
        "marker_sets": str(args.marker_sets.resolve()),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
