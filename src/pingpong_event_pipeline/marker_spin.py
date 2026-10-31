"""Offline finite-rotation fits for the consecutive YOLO marker-track export.

The input is a tracked *cluster centre*, not a calibrated physical marker.
Candidate RPM therefore remains separate from validated physical spin_rpm.
Camera coordinates: x right, y down, z away; visible sphere has negative z.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .spin import _rotate_vectors, solve_constant_sphere_rotation
from .spin_direction import describe_omega, summarize_directions


OBSERVED_SOURCES = {"detection_high", "detection_low", "optical_flow"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tracks(path, dataset, max_gap_us=750, sphere_radius_multiplier=1.0):
    """Validate identities, continuity, crop geometry and manifest time alignment.

    Never guess a frame rate, silently join runs, or accept prediction holds.
    Empty exports are valid and produce an empty report.
    """
    path, dataset = Path(path), Path(dataset)
    multiplier = float(sphere_radius_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("sphere_radius_multiplier must be finite and positive")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups, sources, segments = {}, {}, defaultdict(list)
    for row in rows:
        camera, group, segment = row["camera"], row["group"], row["segment_id"]
        if camera not in {"F3E", "F59"} or Path(group).name != group or not group.endswith("_" + camera):
            raise ValueError("invalid camera/group")
        key = (camera, group)
        if key not in groups:
            folder = dataset / "unlabeled_inference" / camera / group
            meta_path, manifest_path = folder / "metadata.json", folder / "manifest.csv"
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["camera"] != camera or metadata.get("group", group) != group:
                raise ValueError("metadata camera/group mismatch")
            scale = float(metadata["crop_radius_scale"])
            if not np.isfinite(scale) or scale < 1:
                raise ValueError("missing/invalid crop radius scale")
            with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
                manifest_rows = list(csv.DictReader(stream))
            manifest = {int(r["frame_index"]): r for r in manifest_rows}
            if not manifest or len(manifest) != len(manifest_rows):
                raise ValueError("empty/duplicate manifest frames")
            with Image.open(folder / manifest_rows[0]["image"]) as image:
                width, height = image.size
            if width != height or width != 256:
                raise ValueError("expected square 256-pixel YOLO crops")
            groups[key] = {"metadata": metadata, "manifest": manifest,
                           "crop_center_px": width / 2, "sphere_radius_px": width / (2 * scale) * multiplier}
            sources[f"{camera}/{group}"] = {
                "metadata_sha256": sha256(meta_path), "manifest_sha256": sha256(manifest_path),
                "crop_center_px": [width / 2, height / 2], "sphere_radius_px": width / (2 * scale) * multiplier,
                "nominal_sphere_radius_px": width / (2 * scale), "sphere_radius_multiplier": multiplier,
                "crop_radius_scale": scale, "timestamp_source": "manifest_timestamp_us",
                "timing_verified": False, "ball_geometry_verified": False,
                "source_warning": metadata.get("warning"),
            }
        frame, timestamp = int(row["frame_index"]), int(row["timestamp_us"])
        manifest = groups[key]["manifest"]
        if frame not in manifest:
            raise ValueError("track frame missing from manifest")
        original = manifest[frame]
        if (int(original["timestamp_us"]) != timestamp
                or Path(original["image"]).name != Path(row["image"]).name
                or original.get("track_support") != row["track_support"]):
            raise ValueError("track/manifest mismatch")
        if row["source"] not in OBSERVED_SOURCES:
            raise ValueError("non-observed point in rotation export")
        point = np.array([float(row["center_x_px"]), float(row["center_y_px"])])
        xy = (point - groups[key]["crop_center_px"]) / groups[key]["sphere_radius_px"]
        if not np.isfinite(xy).all() or float(xy @ xy) >= 1:
            raise ValueError("point outside visible unit sphere")
        sphere = np.r_[xy, -np.sqrt(1 - xy @ xy)]
        segments[(camera, group, segment)].append({
            **row, "frame": frame, "timestamp": timestamp, "sphere": sphere, "point": point,
            "radius": groups[key]["sphere_radius_px"],
        })
    for key, run in segments.items():
        run.sort(key=lambda row: int(row["segment_frame_offset"]))
        if (len(run) < 3 or any(int(r["segment_length"]) != len(run) for r in run)
                or [int(r["segment_frame_offset"]) for r in run] != list(range(len(run)))
                or len({r["track_id"] for r in run}) != 1):
            raise ValueError(f"invalid consecutive segment: {key}")
        for a, b in zip(run, run[1:]):
            if b["frame"] != a["frame"] + 1 or not 0 < b["timestamp"] - a["timestamp"] <= max_gap_us:
                raise ValueError("nonconsecutive segment / invalid timestamp gap")
    return segments, sources, len(rows)


def make_pairs(runs, first_us, last_us, maximum_pair_us=2000):
    """Use finite lags within one segment; a fallback cannot bridge good points."""
    pairs = []
    for segment, run in runs:
        for i, a in enumerate(run):
            if not first_us <= a["timestamp"] < last_us:
                continue
            for lag in (1, 2, 4):
                if i + lag >= len(run):
                    continue
                b = run[i + lag]
                dt = b["timestamp"] - a["timestamp"]
                if b["timestamp"] >= last_us or dt > maximum_pair_us:
                    continue
                if any(r["track_support"] != "ring_supported" for r in run[i:i + lag + 1]):
                    continue
                pairs.append((segment, a, b, dt * 1e-6))
    return pairs


def fit_pairs(pairs, config):
    """RANSAC + robust finite SO(3) fit, followed by explicit numerical gates."""
    result = {"pairs": len(pairs), "spin_rpm": None, "accepted": False,
              "candidate_spin_rpm": None, "candidate_rps": None, "candidate_axis_camera": None,
              "numerically_usable": False, "rejection_reasons": []}
    result["observation_timestamps_us"] = sorted({r["timestamp"] for _, a, b, _ in pairs for r in (a, b)})
    result["observation_start_us"] = min(result["observation_timestamps_us"], default=None)
    result["observation_end_us"] = max(result["observation_timestamps_us"], default=None)
    minimum = int(config.get("minimum_pairs", 8))
    unique = {(segment, r["frame"]) for segment, a, b, _ in pairs for r in (a, b)}
    result["unique_observations"] = len(unique)
    result["segments"] = len({p[0] for p in pairs})
    if len(pairs) < minimum or len(unique) < int(config.get("minimum_observations", 6)):
        result["rejection_reasons"] = ["insufficient_observations"]
        return result
    s0 = np.array([p[1]["sphere"] for p in pairs])
    s1 = np.array([p[2]["sphere"] for p in pairs])
    dt = np.array([p[3] for p in pairs])
    threshold = float(config.get("angular_threshold_rad", .035))
    omega, inliers, rms, condition = solve_constant_sphere_rotation(
        s0, s1, dt, minimum_correspondences=minimum,
        angular_threshold_rad=threshold, hypotheses=int(config.get("hypotheses", 512)),
        maximum_rpm=float(config.get("maximum_rpm", 12000)), seed=97)
    if not np.isfinite([rms, condition, *omega]).all():
        result["rejection_reasons"] = ["robust_fit_failed"]
        return result
    predicted = _rotate_vectors(s0, omega, dt)
    error_px = np.linalg.norm(predicted[:, :2] - s1[:, :2], axis=1) * np.array([p[1]["radius"] for p in pairs])
    angles = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1, 1))
    magnitude = float(np.linalg.norm(omega))
    rpm = magnitude * 60 / (2 * np.pi)
    fraction = float(np.mean(inliers))
    # Repeated lags are correlated: counts are diagnostics, not independent trials.
    result.update(raw_fit_rpm=rpm, omega_camera_rad_s=omega.tolist(),
                  inlier_pairs=int(inliers.sum()), inlier_fraction=fraction,
                  angular_rms_rad=float(rms), condition=float(condition),
                  median_all_angular_error_rad=float(np.median(angles)),
                  median_all_reprojection_error_px=float(np.median(error_px)),
                  inlier_reprojection_rms_px=float(np.sqrt(np.mean(error_px[inliers] ** 2))),
                  maximum_pair_dt_us=float(dt.max() * 1e6),
                  duration_us=max(p[2]["timestamp"] for p in pairs) - min(p[1]["timestamp"] for p in pairs))
    checks = {
        "low_inlier_fraction": fraction < float(config.get("minimum_inlier_fraction", .70)),
        "ill_conditioned_geometry": condition > float(config.get("maximum_condition", 15)),
        "high_all_point_residual": np.median(angles) > threshold,
        "high_reprojection_error": result["inlier_reprojection_rms_px"] > float(config.get("maximum_rms_px", 3)),
        "insufficient_time_span": result["duration_us"] < int(config.get("minimum_duration_us", 2000)),
        "rotation_alias_risk": magnitude * dt.max() >= np.pi * .8,
    }
    result["rejection_reasons"] = [name for name, failed in checks.items() if failed]
    if not result["rejection_reasons"]:
        # An unresolved axis near zero is different from missing observations.
        stationary_angles = np.arccos(np.clip(np.sum(s0 * s1, axis=1), -1, 1))
        resolution = 2 * threshold / (2 * np.pi * result["duration_us"] * 1e-6)
        near_zero = rpm / 60 <= resolution and float(np.mean(stationary_angles <= threshold)) >= float(config.get("minimum_inlier_fraction", .70))
        result.update(numerically_usable=True, candidate_spin_rpm=rpm, candidate_rps=rpm / 60,
                      candidate_axis_camera=(omega / magnitude).tolist() if magnitude > 1e-8 and not near_zero else None,
                      rotation_signal="near_zero_unresolved_direction" if near_zero else "resolved_rotation_candidate",
                      near_zero_resolution_rps=resolution,
                      stationary_inlier_fraction=float(np.mean(stationary_angles <= threshold)))
        result.update(describe_omega(np.zeros(3) if near_zero else omega))
    return result


def estimate_tracks(tracks, dataset, config):
    segments, sources, count = read_tracks(tracks, dataset, int(config.get("max_gap_us", 750)))
    groups = defaultdict(list)
    for (camera, group, segment), run in segments.items():
        groups[(camera, group)].append((segment, run))
    windows = []
    window_us, step_us = int(config.get("window_us", 20000)), int(config.get("step_us", 10000))
    if not 0 < step_us <= window_us:
        raise ValueError("require 0 < step_us <= window_us")
    for (camera, group), runs in sorted(groups.items()):
        first = min(r["timestamp"] for _, run in runs for r in run)
        last = max(r["timestamp"] for _, run in runs for r in run)
        for start in range(first, last + 1, step_us):
            pairs = make_pairs(runs, start, start + window_us, int(config.get("maximum_pair_us", 2000)))
            fit = fit_pairs(pairs, config)
            fit.update(camera=camera, group=group, start_us=start, end_us=min(start + window_us, last + 1),
                       window_id=f"{group}/{start}", seconds_since_first=(start - first) * 1e-6)
            windows.append(fit)
    usable = [w for w in windows if w["numerically_usable"]]
    group_summaries = []
    for camera, group in sorted(groups):
        subset = [w for w in windows if (w["camera"], w["group"]) == (camera, group)]
        rates = [w["candidate_spin_rpm"] for w in subset if w["numerically_usable"]]
        group_summaries.append({"camera": camera, "group": group, "windows": len(subset),
                                "usable_windows": len(rates), "spin_rpm": None,
                                "candidate_median_rpm": float(np.median(rates)) if rates else None,
                                "candidate_min_rpm": float(min(rates)) if rates else None,
                                "candidate_max_rpm": float(max(rates)) if rates else None,
                                **summarize_directions(subset),
                                "note": "Descriptive summary of overlapping windows, not a confidence interval or stereo fusion."})
    return {"status": "conditional_spin_candidates" if usable else "no_usable_spin_candidates",
            "method": "orthographic_visible_sphere_constant_finite_rotation_ransac_soft_l1",
            "spin_rpm": None, "accepted": False, "physical_measurement_validated": False,
            "input_tracks_sha256": sha256(tracks), "input_observations": count,
            "input_segments": len(segments), "groups": len(groups), "config": config,
            "window_count": len(windows), "usable_window_count": len(usable),
            "rejection_counts": dict(Counter(r for w in windows for r in w["rejection_reasons"])),
            "coordinate_frame": "camera-local: x right, y down, z away; front hemisphere z < 0",
            "assumptions": ["Orthographic projection with source crop_radius_scale, no lens correction.",
                            "Each tracked cluster centre is a fixed physical surface point during a segment.",
                            "Angular velocity is constant within each short window.",
                            "Manifest microseconds represent physical event time; not video playback FPS."],
            "limitations": ["Candidate values are conditional estimates, not validated physical RPM.",
                            "Ball centres/radii, SDK timebase and physical marker identities are unverified.",
                            "Kalman-smoothed bounding-box centres and overlapping event windows can bias spin.",
                            "Overlapping windows/lags are correlated; no statistical confidence interval is claimed.",
                            "Camera axes differ. No stereo fusion without extrinsics and verified synchronization.",
                            "No independent spin ground truth; a low residual alone does not validate the model."],
            "sources": sources, "group_summaries": group_summaries, "windows": windows}


def write_outputs(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "spin_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    group_fields = ["camera", "group", "windows", "usable_windows", "candidate_median_rpm", "candidate_min_rpm", "candidate_max_rpm", "engineering_direction_status", "engineering_candidate_rps", "engineering_candidate_rpm", "stable_windows", "axis_x", "axis_y", "axis_z", "front_motion_label", "image_z_spin_sense", "world_spin_type", "spin_rpm", "note"]
    with (output / "spin_by_group.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=group_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["group_summaries"])
    fields = ["camera", "group", "window_id", "start_us", "end_us", "seconds_since_first",
              "numerically_usable", "candidate_spin_rpm", "candidate_rps", "spin_rpm", "accepted",
              "pairs", "unique_observations", "segments", "inlier_fraction", "condition",
              "inlier_reprojection_rms_px", "rejection_reasons"]
    fields += ["axis_x", "axis_y", "axis_z", "front_direction_dx", "front_direction_dy", "front_motion_label", "image_z_spin_sense", "world_spin_type"]
    fields += ["rotation_signal", "near_zero_resolution_rps", "stationary_inlier_fraction"]
    with (output / "spin_windows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["windows"]:
            writer.writerow({**row, "rejection_reasons": ";".join(row["rejection_reasons"])})
