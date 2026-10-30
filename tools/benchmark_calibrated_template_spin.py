"""Synthetic end-to-end benchmark for calibrated-template 3-D spin recovery."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.stereo_spin import (
    SphereTemplatePoseTracker, estimate_spin_from_template_poses)


def random_template(rng: np.random.Generator, count: int) -> np.ndarray:
    points = []
    while len(points) < count:
        candidate = rng.normal(size=3)
        candidate /= np.linalg.norm(candidate)
        if all(np.linalg.norm(candidate - existing) > .22 for existing in points):
            points.append(candidate)
    return np.asarray(points)


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    vector, _ = cv2.Rodrigues(first @ second.T)
    return float(np.linalg.norm(vector))


def run_case(template: np.ndarray, rpm: float, frames: int, step_us: int,
             noise: float, dropout: float, outliers: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    axis = np.array([.72, -.31, .48], np.float64)
    axis /= np.linalg.norm(axis)
    omega = axis * (rpm * 2 * np.pi / 60.0)
    initial = cv2.Rodrigues(np.array([.35, -.22, .18]))[0]
    timestamps = np.arange(frames, dtype=np.int64) * step_us
    tracker = SphereTemplatePoseTracker(
        template, signature_tolerance=.028, inlier_chord_distance=.035,
        minimum_markers=4, maximum_hypotheses=3500)
    recovered_timestamps = []
    recovered_poses = []
    pose_errors = []
    inlier_counts = []
    hypotheses = 0
    global_frames = local_frames = 0
    global_times_ms = []
    local_times_ms = []
    started = time.perf_counter()
    for timestamp in timestamps:
        truth = cv2.Rodrigues(omega * (timestamp * 1e-6))[0] @ initial
        transformed = (truth @ template.T).T
        visible = np.flatnonzero(transformed[:, 2] < .30)
        retained = visible[rng.random(len(visible)) >= dropout]
        if len(retained) < 5:
            retained = visible[:max(5, len(visible))]
        observed = transformed[retained] + rng.normal(0, noise, (len(retained), 3))
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        if outliers:
            bad = rng.normal(size=(outliers, 3))
            bad /= np.linalg.norm(bad, axis=1, keepdims=True)
            observed = np.vstack((observed, bad))
        observed = observed[rng.permutation(len(observed))]
        frame_started = time.perf_counter()
        pose, template_indices, _, residual, mode, tested = tracker.update(timestamp, observed)
        frame_ms = (time.perf_counter() - frame_started) * 1000
        hypotheses += tested
        global_frames += mode == "global"
        local_frames += mode == "local"
        (global_times_ms if mode == "global" else local_times_ms).append(frame_ms)
        if len(template_indices) < 5 or residual > .025:
            continue
        recovered_timestamps.append(timestamp)
        recovered_poses.append(pose)
        pose_errors.append(rotation_distance(pose, truth))
        inlier_counts.append(len(template_indices))
    spin = estimate_spin_from_template_poses(
        np.asarray(recovered_timestamps), np.asarray(recovered_poses),
        minimum_pairs=8, minimum_span_s=.012, maximum_gap_us=2 * step_us,
        inlier_threshold_rad=.045, maximum_rms_rad=.025,
        minimum_axis_resultant=.90)
    elapsed = time.perf_counter() - started
    rpm_error = abs(spin.rpm - rpm) if spin.rpm is not None else None
    axis_error_deg = (float(np.degrees(np.arccos(np.clip(spin.axis @ axis, -1, 1))))
                      if spin.accepted else None)
    return {
        "truth_rpm": rpm,
        "truth_omega_rad_s": omega.tolist(),
        "input_frames": frames,
        "recovered_pose_frames": len(recovered_poses),
        "pose_fraction": len(recovered_poses) / frames,
        "pose_error_median_deg": (float(np.degrees(np.median(pose_errors)))
                                  if pose_errors else None),
        "template_inliers_median": (float(np.median(inlier_counts))
                                    if inlier_counts else None),
        "estimated_rpm": spin.rpm,
        "rpm_absolute_error": rpm_error,
        "rpm_relative_error": rpm_error / rpm if rpm_error is not None else None,
        "axis_error_deg": axis_error_deg,
        "spin_inlier_pairs": spin.inlier_pairs,
        "spin_residual_rms_rad": spin.residual_rms_rad,
        "axis_resultant": spin.axis_resultant,
        "accepted": spin.accepted,
        "reason": spin.reason,
        "hypotheses_tested": hypotheses,
        "global_match_frames": global_frames,
        "local_track_frames": local_frames,
        "global_match_milliseconds_median": (float(np.median(global_times_ms))
                                             if global_times_ms else None),
        "local_track_milliseconds_median": (float(np.median(local_times_ms))
                                            if local_times_ms else None),
        "local_track_milliseconds_p95": (float(np.percentile(local_times_ms, 95))
                                         if local_times_ms else None),
        "runtime_seconds": elapsed,
        "milliseconds_per_input_frame": elapsed * 1000 / frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="已标定非对称球面模板到三维转速的合成端到端基准")
    parser.add_argument("--rpm", type=float, nargs="+", default=[500, 2000, 6000, 12000])
    parser.add_argument("--frames", type=int, default=31)
    parser.add_argument("--step-us", type=int, default=1000)
    parser.add_argument("--markers", type=int, default=20)
    parser.add_argument("--noise", type=float, default=.002)
    parser.add_argument("--dropout", type=float, default=.12)
    parser.add_argument("--outliers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = random_template(np.random.default_rng(8128), args.markers)
    cases = [run_case(template, rpm, args.frames, args.step_us, args.noise,
                      args.dropout, args.outliers, 9000 + index)
             for index, rpm in enumerate(args.rpm)]
    report = {
        "purpose": "synthetic_calibrated_asymmetric_template_to_3d_spin_benchmark",
        "real_ball_validation": False,
        "parameters": {
            "frames": args.frames, "step_us": args.step_us, "markers": args.markers,
            "surface_direction_noise_std": args.noise, "dropout_fraction": args.dropout,
            "outliers_per_frame": args.outliers,
        },
        "all_cases_accepted": all(case["accepted"] for case in cases),
        "maximum_rpm_relative_error": max(
            (case["rpm_relative_error"] for case in cases
             if case["rpm_relative_error"] is not None), default=None),
        "maximum_axis_error_deg": max(
            (case["axis_error_deg"] for case in cases
             if case["axis_error_deg"] is not None), default=None),
        "cases": cases,
        "interpretation": (
            "This validates the software chain when a correct physical marker template is supplied. "
            "It does not validate the current real ball template or real-camera marker observations."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
