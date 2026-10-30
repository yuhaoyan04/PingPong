"""Ball translation from source manifests, with explicit units and observability.

Pixel velocity is observable without intrinsics. The diameter-scaled velocity
is conditional on a 40 mm sphere and local orthographic projection; it is NOT
the full 3D ball speed. Stereo candidates require actual calibration input.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def local_velocity(timestamps_us, positions, valid=None, max_gap_us=750, window_us=5000):
    """Robust local polynomial derivative, never across invalid samples/gaps."""
    t = np.asarray(timestamps_us, np.int64)
    p = np.asarray(positions, float)
    if p.ndim != 2 or len(p) != len(t) or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError("positions require strictly increasing matching timestamps")
    valid = np.isfinite(p).all(axis=1) if valid is None else np.asarray(valid, bool) & np.isfinite(p).all(axis=1)
    if valid.shape != t.shape or max_gap_us <= 0 or window_us <= 0:
        raise ValueError("invalid derivative configuration")
    velocity = np.full_like(p, np.nan)
    indices = np.flatnonzero(valid)
    breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(t[indices]) > max_gap_us)) + 1
    for run in np.split(indices, breaks):
        for i in run:
            near = run[np.abs(t[run] - t[i]) <= window_us / 2]
            if len(near) < 3:
                continue
            # Millisecond-scaled time keeps quadratic design matrices stable.
            x = (t[near] - t[i]).astype(float) / 1000
            design = np.vander(x, 3 if len(near) >= 5 else 2, increasing=True)
            y = p[near] - p[i]
            weights = np.ones(len(near))
            for _ in range(3):
                root = np.sqrt(weights)
                coef = np.linalg.lstsq(design * root[:, None], y * root[:, None], rcond=None)[0]
                residual = np.linalg.norm(design @ coef - y, axis=1)
                scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-8)
                weights = 1 / (1 + (residual / (2.5 * scale)) ** 2)
            velocity[i] = coef[1] * 1000
    return velocity


def ball_motion(dataset, ball_diameter_m=.040, max_gap_us=750, derivative_window_us=5000):
    if not np.isfinite(ball_diameter_m) or ball_diameter_m <= 0:
        raise ValueError("ball diameter must be positive")
    records, groups = [], []
    for folder in sorted((Path(dataset) / "unlabeled_inference").glob("*/*")):
        if not (folder / "manifest.csv").is_file():
            continue
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        with (folder / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            source = list(csv.DictReader(stream))
        if not source:
            continue
        t = np.array([int(r["timestamp_us"]) for r in source], np.int64)
        centers = np.array([[float(r["center_x_px"]), float(r["center_y_px"])] for r in source])
        radii = np.array([float(r["radius_px"]) for r in source])
        valid = np.array([r.get("track_support") == "ring_supported" for r in source]) & np.isfinite(radii) & (radii > 0)
        # A skipped source frame is a boundary even if its timestamp gap is small.
        frame_indices = np.array([int(r["frame_index"]) for r in source])
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("source frame indices must increase")
        velocity = np.full_like(centers, np.nan)
        breaks = np.flatnonzero(np.diff(frame_indices) != 1) + 1
        for run in np.split(np.arange(len(t)), breaks):
            velocity[run] = local_velocity(t[run], centers[run], valid[run], max_gap_us, derivative_window_us)
        current = []
        for i, row in enumerate(source):
            usable = bool(np.isfinite(velocity[i]).all())
            speed = float(np.linalg.norm(velocity[i])) if usable else None
            current.append({"camera": meta["camera"], "group": folder.name,
                            "frame_index": int(row["frame_index"]), "timestamp_us": int(t[i]),
                            "seconds_since_group_start": float((t[i] - t[0]) * 1e-6),
                            "image": Path(row["image"]).name, "center_x_px": float(centers[i, 0]),
                            "center_y_px": float(centers[i, 1]), "radius_px": float(radii[i]),
                            "track_support": row.get("track_support"), "velocity_usable": usable,
                            "vx_px_s": float(velocity[i, 0]) if usable else None,
                            "vy_px_s": float(velocity[i, 1]) if usable else None,
                            "image_speed_px_s": speed,
                            "projected_speed_ball_diameters_s": speed / (2 * radii[i]) if usable else None,
                            "candidate_transverse_speed_mps": speed * ball_diameter_m / (2 * radii[i]) if usable else None,
                            "speed_mps": None, "physical_measurement_validated": False})
        good = [r for r in current if r["velocity_usable"]]
        groups.append({"camera": meta["camera"], "group": folder.name, "frames": len(current),
                       "velocity_samples": len(good),
                       "image_speed_median_px_s": float(np.median([r["image_speed_px_s"] for r in good])) if good else None,
                       "candidate_transverse_speed_median_mps": float(np.median([r["candidate_transverse_speed_mps"] for r in good])) if good else None,
                       "speed_mps": None, "source_timebase_verified_for_this_recording": False})
        records.extend(current)
    return records, groups


def stereo_motion(records, calibration_path, master_camera, slave_offset_us=0, max_gap_us=750):
    """Optional calibrated 3D candidate; K1/T must describe the explicit master.

    Translation in the input NPZ is interpreted in metres, as produced by this
    project's calibrator. A legacy sync report is not transferred to this data.
    """
    if master_camera not in {"F3E", "F59"}:
        raise ValueError("explicit master camera is required for stereo calibration")
    from .stereo import StereoCalibration
    calibration = StereoCalibration(calibration_path)
    slave_camera = "F59" if master_camera == "F3E" else "F3E"
    groups = defaultdict(dict)
    for record in records:
        serve = record["group"].rsplit("_", 1)[0]
        groups[serve].setdefault(record["camera"], []).append(record)
    output = []
    for serve, cameras in sorted(groups.items()):
        if master_camera not in cameras or slave_camera not in cameras:
            continue
        slave = cameras[slave_camera]
        slave_t = np.array([r["timestamp_us"] + slave_offset_us for r in slave], np.int64)
        flight = []
        for a in cameras[master_camera]:
            time = a["timestamp_us"]
            j = int(np.searchsorted(slave_t, time))
            if j < len(slave) and slave_t[j] == time:
                neighbors = [j]
                bxy = np.array([slave[j]["center_x_px"], slave[j]["center_y_px"]])
            elif 0 < j < len(slave) and slave_t[j] - slave_t[j - 1] <= max_gap_us and slave[j]["frame_index"] == slave[j-1]["frame_index"] + 1:
                neighbors = [j - 1, j]
                alpha = (time - slave_t[j-1]) / (slave_t[j] - slave_t[j-1])
                bxy = (1 - alpha) * np.array([slave[j-1]["center_x_px"], slave[j-1]["center_y_px"]]) + alpha * np.array([slave[j]["center_x_px"], slave[j]["center_y_px"]])
            else:
                continue
            if a["track_support"] != "ring_supported" or any(slave[k]["track_support"] != "ring_supported" for k in neighbors):
                continue
            axy = np.array([a["center_x_px"], a["center_y_px"]])
            if calibration.symmetric_epipolar_error(axy, bxy) > 4:
                continue
            xyz, error = calibration.triangulate(axy, bxy)
            slave_xyz = calibration.R @ xyz + calibration.T.reshape(3)
            if not np.isfinite(xyz).all() or xyz[2] <= 0 or slave_xyz[2] <= 0 or error > 3:
                continue
            flight.append({"serve": serve, "timestamp_us": time, "frame_index": a["frame_index"],
                           "x_m": float(xyz[0]), "y_m": float(xyz[1]), "z_m": float(xyz[2]),
                           "reprojection_error_px": float(error), "master_camera": master_camera,
                           "candidate_3d_speed_mps": None, "speed_mps": None,
                           "physical_measurement_validated": False})
        if flight:
            t = np.array([r["timestamp_us"] for r in flight])
            xyz = np.array([[r["x_m"], r["y_m"], r["z_m"]] for r in flight])
            velocity = np.full_like(xyz, np.nan)
            breaks = np.flatnonzero(np.diff([r["frame_index"] for r in flight]) != 1) + 1
            for run in np.split(np.arange(len(flight)), breaks):
                velocity[run] = local_velocity(t[run], xyz[run], max_gap_us=max_gap_us)
            for r, v in zip(flight, velocity):
                if np.isfinite(v).all():
                    r["candidate_3d_speed_mps"] = float(np.linalg.norm(v))
            output.extend(flight)
    return output


def write_csv(path, records, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or (list(records[0]) if records else ["status"]))
        writer.writeheader()
        writer.writerows(records)
