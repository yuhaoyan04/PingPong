"""Per-serve, time-aligned two-view rotation fitting.

No cross-camera track IDs are invented: independent surface correspondences
constrain ONE angular velocity after rotating both cameras into a common frame.
An experimental relative orientation is cross-validated by holding out whole
serves. It is not a substitute for a measured camera calibration.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .marker_spin import read_tracks, make_pairs, fit_pairs, sha256
from .spin import _rotate_vectors, solve_constant_sphere_rotation
from .spin_direction import describe_omega, summarize_directions

CAMERAS = ("F3E", "F59")


def proper_rotation(value):
    r = np.asarray(value, float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("camera mapping must be a finite proper rotation (det=+1)")
    return r


def align_axes(samples):
    """Signed vector registration; never flip opposite spin to agree."""
    a = np.array([s["axis_F3E"] for s in samples], float)
    b = np.array([s["axis_F59"] for s in samples], float)
    if len(samples) < 3:
        return {"usable": False, "reason": "fewer_than_three_independent_serves"}
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    sa, sb = np.linalg.svd(a, compute_uv=False), np.linalg.svd(b, compute_uv=False)
    spread = float(min(sa[1] / sa[0], sb[1] / sb[0]))
    if spread < .08:
        return {"usable": False, "reason": "axes_too_parallel_to_determine_camera_rotation", "axis_spread_ratio": spread}
    initial, _ = Rotation.align_vectors(a, b)
    fit = least_squares(lambda x: (Rotation.from_rotvec(x).apply(b) - a).ravel(), initial.as_rotvec(),
                        loss="soft_l1", f_scale=.10)
    r = Rotation.from_rotvec(fit.x)
    angles = np.degrees(np.arccos(np.clip(np.sum(a * r.apply(b), axis=1), -1, 1)))
    return {"usable": bool(np.max(angles) <= 20), "reason": "experimental_axis_registration",
            "R_F59_to_F3E": r.as_matrix().tolist(), "axis_spread_ratio": spread,
            "training_serves": [s["serve"] for s in samples], "training_angle_errors_deg": angles.tolist()}


def cross_validate_orientation(samples):
    result = align_axes(samples)
    validation = []
    for sample in samples:
        held = align_axes([s for s in samples if s["serve"] != sample["serve"]])
        angle = None
        if held["usable"]:
            a, b = np.array(sample["axis_F3E"]), np.array(sample["axis_F59"])
            angle = float(np.degrees(np.arccos(np.clip(a @ np.array(held["R_F59_to_F3E"]) @ b, -1, 1))))
        validation.append({"held_out_serve": sample["serve"], "angle_error_deg": angle, "registration": held})
    angles = [v["angle_error_deg"] for v in validation if v["angle_error_deg"] is not None]
    passed = len(angles) == len(samples) and len(angles) >= 4 and np.median(angles) <= 15 and max(angles) <= 25
    return {**result, "usable": bool(result["usable"] and passed), "leave_one_serve_out_passed": bool(passed),
            "leave_one_serve_out": validation, "samples": samples,
            "measured_extrinsics": False, "physical_accuracy_validated": False}


def pair_span(pairs):
    if not pairs:
        return None
    return [min(p[1]["timestamp"] for p in pairs), max(p[2]["timestamp"] for p in pairs)]


def common_support(pairs_by_camera, minimum_span_us=2000):
    spans = [pair_span(pairs_by_camera[c]) for c in CAMERAS]
    if any(s is None for s in spans):
        return {"usable": False, "reason": "one_view_has_no_supported_correspondences", "span_us": 0}
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    observed = [{r["timestamp"] for _, a, b, _ in pairs_by_camera[c] for r in (a, b)} for c in CAMERAS]
    common = len(observed[0] & observed[1])
    return {"usable": bool(hi - lo >= minimum_span_us and common >= 3),
            "reason": "shared_observation_times" if hi - lo >= minimum_span_us and common >= 3 else "insufficient_simultaneous_observations",
            "start_us": lo, "end_us": hi, "span_us": max(0, hi - lo), "shared_timestamps": common}


def fit_two_view(pairs_by_camera, R_F59_to_F3E, config):
    """Balanced robust finite rotation, scored in each original camera view."""
    r = proper_rotation(R_F59_to_F3E)
    support = common_support(pairs_by_camera, int(config.get("minimum_duration_us", 2000)))
    result = {"numerically_usable": False, "candidate_rps": None, "candidate_spin_rpm": None,
              "candidate_axis_camera": None, "spin_rpm": None, "accepted": False,
              "rejection_reasons": [], "common_support": support, "per_view": {}}
    if not support["usable"]:
        result["rejection_reasons"] = [support["reason"]]
        return result
    views, proposals = {}, [np.zeros(3)]
    for camera, mapping in zip(CAMERAS, [np.eye(3), r]):
        pairs = [p for p in pairs_by_camera[camera] if p[1]["timestamp"] >= support["start_us"] and p[2]["timestamp"] <= support["end_us"]]
        if len(pairs) < 4:
            result["rejection_reasons"] = ["too_few_shared_pairs_in_one_view"]
            return result
        s0 = np.array([p[1]["sphere"] for p in pairs]); s1 = np.array([p[2]["sphere"] for p in pairs])
        dt = np.array([p[3] for p in pairs])
        views[camera] = dict(s0=s0 @ mapping.T, s1=s1 @ mapping.T, local_s1=s1,
                             dt=dt, mapping=mapping, radius=np.array([p[1]["radius"] for p in pairs]), pairs=pairs)
        solo = fit_pairs(pairs, config)
        if "omega_camera_rad_s" in solo:
            proposals.append(mapping @ np.array(solo["omega_camera_rad_s"]))
    result["observation_timestamps_us"] = sorted(set.intersection(*[
        {p[1]["timestamp"] for p in v["pairs"]} | {p[2]["timestamp"] for p in v["pairs"]}
        for v in views.values()]))
    # Balance the RANSAC input; a noisy view with many tracks cannot dominate
    # simply by contributing more correlated lags.
    sampled = []
    limit = min(256, min(len(v["dt"]) for v in views.values()))
    for v in views.values():
        indices = np.linspace(0, len(v["dt"]) - 1, limit, dtype=int)
        sampled.append((v["s0"][indices], v["s1"][indices], v["dt"][indices]))
    omega, _, rms, _ = solve_constant_sphere_rotation(
        *[np.concatenate([s[i] for s in sampled]) for i in range(3)],
        minimum_correspondences=int(config.get("minimum_pairs", 8)),
        angular_threshold_rad=float(config.get("angular_threshold_rad", .035)),
        hypotheses=int(config.get("hypotheses", 512)), maximum_rpm=float(config.get("maximum_rpm", 12000)))
    if np.isfinite(rms):
        proposals.append(omega)
    threshold = float(config.get("angular_threshold_rad", .035))

    def evaluate(w):
        angles = [np.arccos(np.clip(np.sum(_rotate_vectors(v["s0"], w, v["dt"]) * v["s1"], axis=1), -1, 1)) for v in views.values()]
        fractions = [float(np.mean(a <= threshold)) for a in angles]
        return (float(np.mean(fractions)), -float(np.mean([np.median(a) for a in angles]))), angles

    omega = max(proposals, key=lambda w: evaluate(w)[0])
    for _ in range(3):
        _, angles = evaluate(omega)
        masks = [a <= threshold for a in angles]
        if any(np.sum(m) < 4 for m in masks):
            break
        def residual(w):
            return np.concatenate([((_rotate_vectors(v["s0"][m], w, v["dt"][m]) - v["s1"][m]) / np.sqrt(np.sum(m))).ravel()
                                   for v, m in zip(views.values(), masks)])
        fitted = least_squares(residual, omega, loss="soft_l1", f_scale=threshold / np.sqrt(max(np.sum(m) for m in masks)), max_nfev=80).x
        if evaluate(fitted)[0][0] + .05 < evaluate(omega)[0][0]:
            break
        omega = fitted
    _, angles = evaluate(omega)
    design, fractions = [], []
    for (camera, v), angular in zip(views.items(), angles):
        inliers = angular <= threshold
        fractions.append(float(np.mean(inliers)))
        predicted_local = _rotate_vectors(v["s0"], omega, v["dt"]) @ v["mapping"]
        px = np.linalg.norm(predicted_local[:, :2] - v["local_s1"][:, :2], axis=1) * v["radius"]
        result["per_view"][camera] = {"pairs": len(angular), "inlier_pairs": int(inliers.sum()),
              "inlier_fraction": fractions[-1], "median_angular_error_rad": float(np.median(angular)),
              "inlier_reprojection_rms_px": float(np.sqrt(np.mean(px[inliers] ** 2))) if np.any(inliers) else None}
        for x, y, z in v["s0"][inliers]:
            design.extend([[0, z, -y], [-z, 0, x], [y, -x, 0]])
    condition = float(np.linalg.cond(design)) if len(design) >= 3 else None
    magnitude = float(np.linalg.norm(omega)); rps = magnitude / (2 * np.pi)
    reasons = []
    if min(fractions) < .35 or max(fractions) < .70 or np.mean(fractions) < .70:
        reasons.append("inconsistent_two_view_surface_motion")
    if any(v["inlier_pairs"] < 4 for v in result["per_view"].values()):
        reasons.append("insufficient_inliers_in_one_view")
    if condition is None or not np.isfinite(condition) or condition > float(config.get("maximum_condition", 15)):
        reasons.append("ill_conditioned_joint_geometry")
    if any(v["inlier_reprojection_rms_px"] is None or v["inlier_reprojection_rms_px"] > float(config.get("maximum_rms_px", 3)) for v in result["per_view"].values()):
        reasons.append("high_joint_reprojection_error")
    if magnitude > float(config.get("maximum_rpm", 12000)) * 2 * np.pi / 60:
        reasons.append("exceeds_configured_maximum")
    if any(magnitude * v["dt"].max() >= .8 * np.pi for v in views.values()):
        reasons.append("rotation_alias_risk")
    result.update(raw_fit_rps=rps, omega_camera_rad_s=omega.tolist(), condition=condition if condition is not None and np.isfinite(condition) else None,
                  balanced_inlier_fraction=float(np.mean(fractions)), rejection_reasons=reasons,
                  reference_camera="F3E", numerically_usable=not reasons)
    if not reasons:
        resolution = 2 * threshold / (2 * np.pi * support["span_us"] * 1e-6)
        near_zero = rps <= resolution and all(float(np.mean(np.arccos(np.clip(np.sum(v["s0"] * v["s1"], axis=1), -1, 1)) <= threshold)) >= .70 for v in views.values())
        result.update(candidate_rps=rps, candidate_spin_rpm=rps * 60,
                      candidate_axis_camera=(omega / magnitude).tolist() if magnitude > 1e-8 and not near_zero else None,
                      rotation_signal="near_zero_unresolved_direction" if near_zero else "resolved_rotation_candidate")
        result.update(describe_omega(np.zeros(3) if near_zero else omega))
    return result


def analyze_serves(tracks, dataset, config, previous_report=None, orientation_profile=None,
                   radius_multiplier=1.0, fixed_relative_orientations=None):
    segments, sources, count = read_tracks(tracks, dataset, int(config.get("max_gap_us", 750)), radius_multiplier,
                                          config.get('trajectory_smoothing'))
    groups = defaultdict(lambda: defaultdict(list))
    for (camera, group, segment), run in segments.items():
        groups[group.rsplit("_", 1)[0]][camera].append((segment, run))
    dataset = Path(dataset)
    work, diagnostics, samples = {}, {}, []
    for serve, runs in sorted(groups.items()):
        views = {}
        for camera in CAMERAS:
            import csv
            folder = dataset / "unlabeled_inference" / camera / f"{serve}_{camera}"
            with (folder / "manifest.csv").open(encoding="utf-8-sig") as stream:
                manifest = list(csv.DictReader(stream))
            times = [int(row["timestamp_us"]) for row in manifest]
            observations = [row for _, run in runs[camera] for row in run]
            views[camera] = {"frames": len(manifest), "start_us": min(times), "end_us": max(times),
                "unsupported_ball_frames": sum(row["track_support"] != "ring_supported" for row in manifest),
                "rotation_observations": len(observations), "segments": len(runs[camera]),
                "observed_frames": len({row["timestamp"] for row in observations}),
                "supported_observed_frames": len({row["timestamp"] for row in observations if row["track_support"] == "ring_supported"}),
                "longest_segment_frames": max((len(run) for _, run in runs[camera]), default=0)}
            views[camera]["supported_pair_span_us"] = pair_span(make_pairs(runs[camera], min(times), max(times)+1,
                                                                                       int(config.get("maximum_pair_us", 2000))))
        first, last = min(v["start_us"] for v in views.values()), max(v["end_us"] for v in views.values())
        rows = []
        for start in range(first, last + 1, int(config.get("step_us", 10000))):
            end = start + int(config.get("window_us", 20000))
            pairs = {c: make_pairs(runs[c], start, end, int(config.get("maximum_pair_us", 2000))) for c in CAMERAS}
            fits = {c: fit_pairs(pairs[c], config) for c in CAMERAS}
            rows.append({"serve": serve, "start_us": start, "end_us": min(end, last + 1), "pairs": pairs, "mono": fits,
                         "common_support": common_support(pairs, int(config.get("minimum_duration_us", 2000)))})
        for camera in CAMERAS:
            valid = [w["mono"][camera] for w in rows if w["mono"][camera]["numerically_usable"]]
            views[camera]["aligned_summary"] = summarize_directions([
                {**w["mono"][camera], "window_id": f"{serve}_{camera}/{w['start_us']}"} for w in rows])
            views[camera]["aligned_windows"] = len(rows)
            views[camera]["aligned_usable_windows"] = len(valid)
            views[camera]["aligned_rejections"] = dict(Counter(z for w in rows for z in w["mono"][camera]["rejection_reasons"]))
            views[camera]["aligned_candidate_rps"] = [w["candidate_rps"] for w in valid]
            if previous_report:
                original = next(g for g in previous_report["group_summaries"] if g["group"] == f"{serve}_{camera}")
                original_windows = [w for w in previous_report["windows"] if w["group"] == original["group"]]
                views[camera]["previous_summary"] = original
                views[camera]["previous_rejections"] = dict(Counter(z for w in original_windows for z in w["rejection_reasons"]))
                views[camera]["previous_usable_rps"] = [w["candidate_rps"] for w in original_windows if w["numerically_usable"]]
        shared = [w for w in rows if w["common_support"]["usable"] and all(w["mono"][c]["numerically_usable"] and w["mono"][c]["candidate_axis_camera"] is not None for c in CAMERAS)]
        if shared:
            sample = {"serve": serve, "shared_windows": len(shared)}
            for c in CAMERAS:
                axis = np.median([w["mono"][c]["candidate_axis_camera"] for w in shared], axis=0)
                sample["axis_" + c] = (axis / np.linalg.norm(axis)).tolist()
            samples.append(sample)
        work[serve], diagnostics[serve] = rows, views
    registration = cross_validate_orientation(samples)
    measured = None
    if orientation_profile:
        profile = json.loads(Path(orientation_profile).read_text())
        mapping = profile["camera_to_reference"]
        measured = proper_rotation(mapping["F3E"]).T @ proper_rotation(mapping["F59"])
    output = []
    for serve, rows in work.items():
        others = [s for s in samples if s["serve"] != serve]
        held = align_axes(others)
        if fixed_relative_orientations is not None:
            fixed = fixed_relative_orientations[serve]
            mapping = proper_rotation(fixed["R_F59_to_F3E"]) if fixed["R_F59_to_F3E"] is not None else None
            mode = "fixed_baseline_orientation_for_radius_comparison"
        elif measured is not None:
            mapping, mode = measured, "provided_camera_orientation"
        elif registration["usable"] and held["usable"]:
            mapping, mode = np.array(held["R_F59_to_F3E"]), "experimental_orientation_target_serve_excluded"
        else:
            mapping, mode = None, "no_reliable_camera_orientation"
        windows = []
        for row in rows:
            fit = fit_two_view(row["pairs"], mapping, config) if mapping is not None else {
                "numerically_usable": False, "candidate_rps": None, "candidate_spin_rpm": None,
                "candidate_axis_camera": None, "rejection_reasons": [mode], "per_view": {}}
            fit.update(serve=serve, window_id=f"{serve}/{row['start_us']}", start_us=row["start_us"], end_us=row["end_us"],
                       mono=row["mono"], aligned_common_support=row["common_support"])
            windows.append(fit)
        summary = summarize_directions(windows)
        usable = [w for w in windows if w["numerically_usable"]]
        output.append({"serve": serve, "views": diagnostics[serve], "orientation_mode": mode,
                       "orientation_training_serves": fixed_relative_orientations[serve]["orientation_training_serves"] if fixed_relative_orientations is not None else (held.get("training_serves", []) if measured is None else []),
                       "R_F59_to_F3E": mapping.tolist() if mapping is not None else None,
                       "windows": windows, "joint_summary": summary, "joint_usable_windows": len(usable),
                       "joint_window_rps_range": [min(w["candidate_rps"] for w in usable), max(w["candidate_rps"] for w in usable)] if usable else None,
                       "joint_rejections": dict(Counter(z for w in windows for z in w["rejection_reasons"]))})
    return {"method": "two_view_shared_finite_SO3_per_serve", "reference_camera": "F3E",
            "config": config, "tracks_sha256": sha256(tracks), "dataset": str(dataset.resolve()),
            "input_observations": count, "sources": sources, "serves": output,
            "sphere_radius_multiplier": radius_multiplier,
            "orientation_fixed_to_baseline": fixed_relative_orientations is not None,
            "orientation_registration": registration, "orientation_profile": str(orientation_profile) if orientation_profile else None,
            "minimum_allowed_rps": 0, "physical_accuracy_validated": False, "training_raw_timing_verified": False,
            "limitations": ["Experimental relative orientation is inferred from other serves, not measured camera extrinsics.",
                            "Each target serve is excluded from its orientation fit; independent camera calibration is still needed.",
                            "Shared manifest timestamps are assumed physical and synchronized; ball RAW timebase remains unverified.",
                            "Cluster centres, orthographic sphere geometry and 1 ms accumulation can bias rotation.",
                            "No cross-camera marker identity is assumed; both views constrain the shared angular velocity.",
                            "No observations is unknown, not zero spin. Near-zero spin has no resolved direction."]}
