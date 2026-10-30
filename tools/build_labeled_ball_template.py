"""Build a body-fixed sphere marker template from reviewed dual-camera IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_event_pipeline.stereo import StereoCalibration
from pingpong_event_pipeline.stereo_spin import (
    audit_sphere_template_symmetry, build_labeled_sphere_template)
from visualize_dual_annotation_pack import load_records


def patch_to_full(point, record, image_size):
    radius = float(record["radius_px_full_image"])
    scale = radius / (0.5 * image_size)
    return np.asarray(point, np.float64) * scale + np.array([
        float(record["center_x_full_image"]) - radius,
        float(record["center_y_full_image"]) - radius,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从双路人工共享物理 ID 三角化并融合真实球面非对称模板")
    parser.add_argument("--dual-annotations", type=Path, required=True)
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True,
                        help="directory containing master_path.npz and slave_path.npz")
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration/stereo_calibration.npz"))
    parser.add_argument("--identity-evaluation", type=Path)
    parser.add_argument("--ball-radius-m", type=float, default=.020)
    parser.add_argument("--maximum-reprojection-px", type=float, default=3.0)
    parser.add_argument("--maximum-shell-error-mm", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True,
                        help="output NPZ; JSON report uses the same stem")
    args = parser.parse_args()

    dual = json.loads(args.dual_annotations.read_text(encoding="utf-8"))
    master_records = load_records(args.master_annotations)
    slave_records = load_records(args.slave_annotations)
    image_size = int(dual["image_size_px"][0])
    calibration = StereoCalibration(args.calibration)
    master_path = np.load(args.paths / "master_path.npz")
    slave_path = np.load(args.paths / "slave_path.npz")
    master_times = master_path["timestamps_us"].astype(np.int64)
    slave_times = slave_path["timestamps_us"].astype(np.int64)
    if not np.array_equal(master_times, slave_times):
        raise ValueError("master/slave circle paths do not share exact timestamps")
    master_centers = master_path["centers_px"].astype(np.float64)
    slave_centers = slave_path["centers_px"].astype(np.float64)
    path_index = {int(timestamp): index for index, timestamp in enumerate(master_times)}

    labeled_frames = []
    total_shared_labels = accepted_triangulations = 0
    reprojection_errors = []
    shell_errors = []
    rejected_missing_path = rejected_geometry = 0
    for frame in dual["frames"]:
        if frame["status"] != "reviewed":
            continue
        timestamp = int(frame["timestamp_us"])
        if timestamp not in path_index or timestamp not in master_records or timestamp not in slave_records:
            rejected_missing_path += 1
            continue
        master_labels = {int(item["id"]): item for item in frame["master_labels"]}
        slave_labels = {int(item["id"]): item for item in frame["slave_labels"]}
        shared = sorted(set(master_labels) & set(slave_labels))
        total_shared_labels += len(shared)
        index = path_index[timestamp]
        center_m, center_error = calibration.triangulate(
            master_centers[index], slave_centers[index])
        directions = {}
        for identifier in shared:
            left = patch_to_full(
                [master_labels[identifier]["x"], master_labels[identifier]["y"]],
                master_records[timestamp], image_size)
            right = patch_to_full(
                [slave_labels[identifier]["x"], slave_labels[identifier]["y"]],
                slave_records[timestamp], image_size)
            point, reprojection = calibration.triangulate(left, right)
            shell = abs(float(np.linalg.norm(point - center_m)) - args.ball_radius_m)
            if (not np.isfinite(point).all() or point[2] <= 0
                    or reprojection > args.maximum_reprojection_px
                    or shell > args.maximum_shell_error_mm * 1e-3):
                rejected_geometry += 1
                continue
            direction = point - center_m
            direction /= max(np.linalg.norm(direction), 1e-12)
            directions[identifier] = direction
            accepted_triangulations += 1
            reprojection_errors.append(reprojection)
            shell_errors.append(shell)
        if directions:
            labeled_frames.append({"timestamp_us": timestamp, "directions": directions,
                                   "center_reprojection_error_px": center_error})

    template = build_labeled_sphere_template(labeled_frames)
    symmetry = audit_sphere_template_symmetry(template.directions)
    geometry_accepted = bool(template.accepted and symmetry["passed"])
    identity_accepted = False
    if args.identity_evaluation and args.identity_evaluation.exists():
        identity_accepted = bool(json.loads(args.identity_evaluation.read_text(
            encoding="utf-8")).get("accepted", False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        marker_ids=template.marker_ids,
        directions=template.directions,
        observation_counts=template.observation_counts,
        frame_indices=template.frame_indices,
        timestamps_us=template.timestamps_us,
        poses=template.poses,
        frame_residuals=template.frame_residuals,
        physical_template_geometry_accepted=np.asarray(geometry_accepted),
    )
    report = {
        "purpose": "human_identified_dual_camera_physical_ball_template_calibration",
        "status": "accepted" if geometry_accepted and identity_accepted else "rejected",
        "physical_template_geometry_accepted": geometry_accepted,
        "template_fusion_accepted": template.accepted,
        "rotational_symmetry_audit": symmetry,
        "candidate_tracking_identity_accepted": identity_accepted,
        "spin_solver_allowed": bool(geometry_accepted and identity_accepted),
        "reason": ("physical template and candidate identity gates passed"
                   if geometry_accepted and identity_accepted else
                   (template.reason if not template.accepted else symmetry["reason"])),
        "reviewed_frames": sum(frame["status"] == "reviewed" for frame in dual["frames"]),
        "total_shared_labels": total_shared_labels,
        "accepted_triangulations": accepted_triangulations,
        "rejected_missing_path_frames": rejected_missing_path,
        "rejected_marker_geometry": rejected_geometry,
        "connected_template_frames": len(template.timestamps_us),
        "template_markers": len(template.marker_ids),
        "markers_with_repeated_observations": int(np.count_nonzero(
            template.observation_counts >= 2)),
        "reprojection_error_px_median": (float(np.median(reprojection_errors))
                                         if reprojection_errors else None),
        "shell_error_mm_median": (float(np.median(shell_errors) * 1000)
                                  if shell_errors else None),
        "frame_template_residual_median": (float(np.median(template.frame_residuals))
                                           if len(template.frame_residuals) else None),
        "calibration_timestamp_start_us": (int(template.timestamps_us[0])
                                           if len(template.timestamps_us) else None),
        "calibration_timestamp_end_us": (int(template.timestamps_us[-1])
                                         if len(template.timestamps_us) else None),
        "template_npz": str(args.output.resolve()),
        "limitations": [
            "the body coordinate orientation is arbitrary and defined by the seed frame",
            "template geometry acceptance does not replace absolute RPM ground-truth validation",
            "spin remains disabled unless the independent candidate identity gate also passes",
        ],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
