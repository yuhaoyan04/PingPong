"""Signed camera-frame rotation directions and temporal engineering checks.

No magnitude clipping or projection onto a literature target. Temporal
consistency is not a claim of accuracy or independent sample statistics.
"""
from __future__ import annotations

import math
import numpy as np


def describe_omega(omega, minimum_component_fraction=.15):
    w = np.asarray(omega, float)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("omega must be a finite camera-frame 3-vector")
    magnitude = float(np.linalg.norm(w))
    result = {"axis_x": None, "axis_y": None, "axis_z": None,
              "front_direction_dx": None, "front_direction_dy": None,
              "front_motion_label": "unresolved", "image_z_spin_sense": "unresolved",
              "world_spin_type": None}
    if magnitude < 1e-8:
        return result
    axis = w / magnitude
    result.update(axis_x=float(axis[0]), axis_y=float(axis[1]), axis_z=float(axis[2]))
    # omega x (0,0,-1) = (-omega_y, omega_x, 0).
    v = np.array([-w[1], w[0]])
    length = float(np.linalg.norm(v))
    if length / magnitude >= minimum_component_fraction:
        v /= length
        labels = ["right", "up_right", "up", "up_left", "left", "down_left", "down", "down_right"]
        angle = math.degrees(math.atan2(-v[1], v[0])) % 360
        result.update(front_direction_dx=float(v[0]), front_direction_dy=float(v[1]),
                      front_motion_label=labels[int(math.floor((angle + 22.5) / 45)) % 8])
    if abs(axis[2]) >= minimum_component_fraction:
        # x right/y down: positive z induces clockwise image-plane rotation.
        result["image_z_spin_sense"] = "clockwise" if axis[2] > 0 else "counterclockwise"
    else:
        result["image_z_spin_sense"] = "small_view_axis_component"
    return result


def summarize_directions(windows, angle_limit_deg=15, relative_rate_limit=.20, minimum_windows=3):
    usable = [w for w in windows if w.get("numerically_usable") and w.get("candidate_axis_camera") is not None]
    result = {"engineering_direction_status": "no_candidate", "engineering_stable": False,
              "direction_windows": len(usable), "stable_windows": 0,
              "engineering_candidate_rps": None, "engineering_candidate_rpm": None,
              "axis_x": None, "axis_y": None, "axis_z": None,
              "front_direction_dx": None, "front_direction_dy": None,
              "front_motion_label": "unresolved", "image_z_spin_sense": "unresolved",
              "world_spin_type": None, "excluded_windows": [],
              "all_windows_max_axis_deviation_deg": None, "stable_windows_max_axis_deviation_deg": None,
              "relative_rate_mad": None, "physical_accuracy_validated": False}
    if not usable:
        return result
    axes = np.array([w["candidate_axis_camera"] for w in usable], float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    rates = np.array([w["candidate_spin_rpm"] / 60 for w in usable])
    distance = np.degrees(np.arccos(np.clip(axes @ axes.T, -1, 1)))
    # Signed medoid: opposite angular velocities remain 180 degrees apart.
    medoid = int(np.argmin(np.median(distance, axis=1)))
    angle = distance[medoid]
    median_rate = float(np.median(rates))
    relative = np.abs(rates - median_rate) / max(median_rate, 1e-9)
    selected = (angle <= angle_limit_deg) & (relative <= relative_rate_limit)
    for i in np.flatnonzero(~selected):
        result["excluded_windows"].append({"window_id": usable[i]["window_id"],
                                           "angle_to_medoid_deg": float(angle[i]),
                                           "relative_rate_difference": float(relative[i]),
                                           "reason": "temporal_direction_or_rate_outlier"})
    required = max(minimum_windows, int(math.ceil(.6 * len(usable))))
    stable = int(selected.sum()) >= required
    result.update(engineering_direction_status="stable_candidate" if stable else
                  ("insufficient_windows" if len(usable) < minimum_windows else "inconsistent_candidate"),
                  stable_windows=int(selected.sum()), engineering_stable=stable,
                  all_windows_max_axis_deviation_deg=float(np.max(angle)),
                  relative_rate_mad=float(np.median(np.abs(rates - median_rate)) / max(median_rate, 1e-9)))
    if stable:
        axis = np.sum(axes[selected], axis=0)
        axis /= np.linalg.norm(axis)
        rps = float(np.median(rates[selected]))
        result.update(describe_omega(axis))
        result.update(engineering_candidate_rps=rps, engineering_candidate_rpm=rps * 60,
                      stable_windows_max_axis_deviation_deg=float(np.max(np.degrees(np.arccos(np.clip(axes[selected] @ axis, -1, 1))))))
    return result


def classify_in_reference(omega, flight_direction, up_direction):
    """Sport labels only with an explicit common reference/flight/up frame.

    Topspin makes the upper ball surface move along flight: its axis is
    up x forward. Camera image-up is not silently substituted for world-up.
    """
    w, forward, up = map(lambda v: np.array(v, dtype=float, copy=True), (omega, flight_direction, up_direction))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (w, forward, up)):
        raise ValueError("reference vectors must be finite 3-vectors")
    if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(up) < 1e-9:
        raise ValueError("flight/up vectors must be nonzero")
    forward /= np.linalg.norm(forward)
    up -= (up @ forward) * forward
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("flight and up cannot be parallel")
    up /= np.linalg.norm(up)
    top_axis = np.cross(up, forward)
    top, side, cork = float(w @ top_axis), float(w @ up), float(w @ forward)
    transverse = float(np.hypot(top, side))
    magnitude = float(np.linalg.norm(w))
    result = {"topspin_component_rad_s": top, "side_component_rad_s": side,
              "corkscrew_component_rad_s": cork, "world_spin_type": None}
    if magnitude < 1e-8:
        return result
    if transverse / magnitude < .15:
        result["world_spin_type"] = "predominantly_corkscrew"
        return result
    angle = math.degrees(math.atan2(side, top)) % 360
    names = ["topspin", "top_side_ccw_from_above", "side_ccw_from_above", "back_side_ccw_from_above",
             "backspin", "back_side_cw_from_above", "side_cw_from_above", "top_side_cw_from_above"]
    result["world_spin_type"] = names[int(math.floor((angle + 22.5) / 45)) % 8]
    return result
