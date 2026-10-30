"""Estimate ball spin by maximising motion-compensated event sharpness.

Each event is lifted from the image disc to the visible unit hemisphere.  A
candidate constant angular velocity rotates every event back to the centre of
its short time window.  The correct velocity aligns marker edges and therefore
maximises the squared mass of a polarity-separated event histogram.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from estimate_event_plane_spin import normalized_events


def warp_points(points: np.ndarray, dt_s: np.ndarray, omega: np.ndarray):
    """Rotate hemisphere points backwards by omega*dt (Rodrigues)."""
    magnitude = float(np.linalg.norm(omega))
    if magnitude < 1e-9:
        return points
    axis = omega / magnitude
    angle = -magnitude * dt_s
    cosine = np.cos(angle)[:, None]
    sine = np.sin(angle)[:, None]
    cross = np.cross(np.broadcast_to(axis, points.shape), points)
    dot = points @ axis
    return (points * cosine + cross * sine
            + dot[:, None] * axis[None, :] * (1.0 - cosine))


def sharpness(omega, points, dt_s, polarity, bins=48, radius=.72):
    warped = warp_points(points, dt_s, np.asarray(omega, np.float64))
    x, y = warped[:, 0], warped[:, 1]
    valid = (np.abs(x) <= radius) & (np.abs(y) <= radius) & (warped[:, 2] > 0)
    if np.count_nonzero(valid) < .90 * len(points):
        return -1e9
    ix = np.floor((x[valid] + radius) * bins / (2 * radius)).astype(np.int32)
    iy = np.floor((y[valid] + radius) * bins / (2 * radius)).astype(np.int32)
    np.clip(ix, 0, bins - 1, out=ix); np.clip(iy, 0, bins - 1, out=iy)
    index = (polarity[valid].astype(np.int64) * bins + iy) * bins + ix
    hist = np.bincount(index, minlength=2 * bins * bins).astype(np.float64)
    # Normalisation prevents candidates winning merely by retaining more events.
    return float(np.sum(hist * hist) / max(1.0, np.sum(hist) ** 2))


def solve_window(x, y, t, polarity, maximum_rpm, bins, seed):
    z = np.sqrt(np.maximum(1e-8, 1.0 - x*x - y*y))
    points = np.column_stack((x, y, z))
    dt = (t.astype(np.float64) - np.median(t)) * 1e-6
    limit = maximum_rpm * 2 * np.pi / 60.0
    baseline = sharpness(np.zeros(3), points, dt, polarity, bins)
    objective = lambda w: -sharpness(w, points, dt, polarity, bins)
    result = differential_evolution(
        objective, [(-limit, limit)] * 3, seed=seed, popsize=8,
        maxiter=24, polish=False, updating="immediate", workers=1)
    omega = result.x
    score = -result.fun
    return omega, score, baseline


def solve_shared(windows, maximum_rpm, bins, seed):
    """Find one angular velocity supported by all usable flight windows."""
    prepared = []
    for x, y, t, polarity in windows:
        z = np.sqrt(np.maximum(1e-8, 1.0 - x*x - y*y))
        points = np.column_stack((x, y, z))
        dt = (t.astype(np.float64) - np.median(t)) * 1e-6
        baseline = sharpness(np.zeros(3), points, dt, polarity, bins)
        prepared.append((points, dt, polarity, baseline))
    limit = maximum_rpm * 2 * np.pi / 60.0
    def objective(omega):
        gains = [sharpness(omega, p, dt, pol, bins) / max(base, 1e-12)
                 for p, dt, pol, base in prepared]
        if any(g <= 0 for g in gains):
            return 1e6
        # A trimmed log mean makes a candidate improve several windows rather
        # than exploit one unusually sparse interval.
        values = np.sort(np.log(np.asarray(gains)))
        if len(values) >= 5:
            values = values[1:-1]
        return -float(np.mean(values))
    result = differential_evolution(
        objective, [(-limit, limit)] * 3, seed=seed, popsize=10,
        maxiter=38, polish=False, updating="immediate", workers=1)
    omega = result.x
    gains = np.asarray([sharpness(omega, p, dt, pol, bins) / max(base, 1e-12)
                        for p, dt, pol, base in prepared])
    return omega, gains


def main():
    parser = argparse.ArgumentParser(description="事件反向旋转聚焦转速估计")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inner-radius", type=float, default=.72)
    parser.add_argument("--window-us", type=int, default=5000)
    parser.add_argument("--stride-us", type=int, default=10000)
    parser.add_argument("--maximum-rpm", type=float, default=12000)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--minimum-events", type=int, default=300)
    parser.add_argument("--maximum-windows", type=int, default=18)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    x, y, t, polarity = normalized_events(args.raw, args.path, args.inner_radius)
    starts = np.arange(int(t.min()), int(t.max()) - args.window_us + 1,
                       args.stride_us, dtype=np.int64)
    if len(starts) > args.maximum_windows:
        select = np.linspace(0, len(starts)-1, args.maximum_windows).round().astype(int)
        starts = starts[select]
    estimates = []
    window_data = []
    for number, start in enumerate(starts):
        take = (t >= start) & (t < start + args.window_us)
        if np.count_nonzero(take) < args.minimum_events:
            continue
        window_data.append((x[take], y[take], t[take], polarity[take]))
        estimates.append({"start_us": int(start), "events": int(np.count_nonzero(take))})
    if window_data:
        consensus, gains = solve_shared(window_data, args.maximum_rpm, args.bins,
                                        args.seed)
    else:
        consensus, gains = np.zeros(3), np.empty(0)
    for record, gain in zip(estimates, gains):
        record["sharpness_gain"] = float(gain)
    credible = gains >= 1.01
    magnitude = float(np.linalg.norm(consensus))
    report = {
        "purpose": "marker_id_free_event_warp_spin_candidate",
        "events_inside_ball": int(len(x)), "windows": estimates,
        "credible_windows": int(np.count_nonzero(credible)),
        "rpm_median_vector": magnitude * 60 / (2*np.pi) if magnitude else None,
        "axis_camera": (consensus/magnitude).tolist() if magnitude else None,
        "median_sharpness_gain": float(np.median(gains)) if len(gains) else None,
        "parameters": vars(args),
        "status": "candidate_requires_dual_camera_and_parameter_stability",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
