from __future__ import annotations

from typing import List

import cv2
import numpy as np


def marker_structure_metrics(events: np.ndarray, reference_timestamp_us: int,
                             center_px, velocity_px_s, radius_px: float,
                             inner_radius_fraction: float = .60,
                             patch_size: int = 64) -> dict:
    """Measure live ball-interior polarity balance and spatial concentration.

    The tracked centre is extrapolated to each event timestamp before the event
    is tested against the inner ball disc.  Metrics are observability hints, not
    physical-marker detections or a spin estimate.
    """
    empty = {"events": 0, "on": 0, "off": 0, "polarity_balance": 0.0,
             "density_peak_to_mean": 0.0}
    if events is None or len(events) == 0 or radius_px <= 0:
        return empty
    timestamps = events["timestamp"].astype(np.int64, copy=False)
    dt = (timestamps - int(reference_timestamp_us)).astype(np.float64) * 1e-6
    cx = float(center_px[0]) + float(velocity_px_s[0]) * dt
    cy = float(center_px[1]) + float(velocity_px_s[1]) * dt
    dx = events["x"].astype(np.float64) - cx
    dy = events["y"].astype(np.float64) - cy
    limit = float(radius_px) * float(inner_radius_fraction)
    inside = dx * dx + dy * dy <= limit * limit
    if not np.any(inside):
        return empty
    polarity = events["polarity"][inside].astype(bool)
    on = int(np.count_nonzero(polarity)); off = int(len(polarity) - on)
    scale = (0.5 * patch_size) / max(float(radius_px), 1.0)
    u = np.rint(dx[inside] * scale + patch_size / 2).astype(np.int64)
    v = np.rint(dy[inside] * scale + patch_size / 2).astype(np.int64)
    valid = (u >= 0) & (u < patch_size) & (v >= 0) & (v < patch_size)
    occupancy = np.zeros((patch_size, patch_size), np.float32)
    occupancy[v[valid], u[valid]] = 1.0
    density = cv2.GaussianBlur(occupancy, (0, 0), max(1.0, patch_size * 5 / 256))
    y, x = np.ogrid[:patch_size, :patch_size]
    mask = np.hypot(x - patch_size / 2, y - patch_size / 2) < (
        .5 * patch_size * inner_radius_fraction)
    values = density[mask]
    concentration = float(np.max(values) / max(float(np.mean(values)), 1e-9))
    return {
        "events": int(np.count_nonzero(inside)), "on": on, "off": off,
        "polarity_balance": min(on, off) / max(on, off, 1),
        "density_peak_to_mean": concentration,
    }


def summarize_spin_segments(
    timestamps_us: np.ndarray,
    omega: np.ndarray,
    quality: np.ndarray,
    disagreement: np.ndarray,
    maximum_gap_us: int = 50_000,
    minimum_samples: int = 20,
) -> List[dict]:
    """Return robust, internally consistent spin estimates for contiguous tracks."""
    timestamps_us = np.asarray(timestamps_us, dtype=np.int64)
    omega = np.asarray(omega, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    valid = (np.isfinite(omega).all(axis=1) & np.isfinite(quality) &
             np.isfinite(disagreement) & (np.linalg.norm(omega, axis=1) > 1e-6))
    timestamps_us, omega = timestamps_us[valid], omega[valid]
    quality, disagreement = quality[valid], disagreement[valid]
    if not len(timestamps_us):
        return []
    boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps_us) > maximum_gap_us) + 1,
                       len(timestamps_us)]
    summaries = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        vectors = omega[begin:end]
        if len(vectors) < minimum_samples:
            continue
        center = np.median(vectors, axis=0)
        for _ in range(4):
            magnitude = np.linalg.norm(vectors, axis=1)
            center_magnitude = np.linalg.norm(center)
            cosine = np.sum(vectors * center, axis=1) / np.maximum(magnitude * center_magnitude, 1e-9)
            angular = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            rpm = magnitude * 60.0 / (2.0 * np.pi)
            rpm_median = np.median(rpm)
            mad = np.median(np.abs(rpm - rpm_median)) + 1e-9
            inliers = (angular <= 30.0) & (np.abs(rpm - rpm_median) <= 3.5 * 1.4826 * mad)
            if np.count_nonzero(inliers) < minimum_samples:
                break
            center = np.average(vectors[inliers], axis=0, weights=np.maximum(quality[begin:end][inliers], 1e-3))
        if np.count_nonzero(inliers) < minimum_samples:
            continue
        accepted = vectors[inliers]
        magnitudes = np.linalg.norm(accepted, axis=1)
        rpm_values = magnitudes * 60.0 / (2.0 * np.pi)
        center_magnitude = float(np.linalg.norm(center))
        axis = center / max(center_magnitude, 1e-9)
        angular = np.degrees(np.arccos(np.clip(
            accepted @ axis / np.maximum(magnitudes, 1e-9), -1.0, 1.0)))
        rpm_mean = float(center_magnitude * 60.0 / (2.0 * np.pi))
        rpm_cv = float(np.std(rpm_values) / max(np.mean(rpm_values), 1e-9))
        angle_p90 = float(np.percentile(angular, 90))
        q = quality[begin:end][inliers]
        d = disagreement[begin:end][inliers]
        consistent = bool(len(accepted) >= minimum_samples and angle_p90 <= 20.0 and
                          rpm_cv <= 0.25 and np.median(d) <= 0.30 and np.median(q) >= 0.25)
        summaries.append({
            "start_timestamp_us": int(timestamps_us[begin:end][inliers][0]),
            "end_timestamp_us": int(timestamps_us[begin:end][inliers][-1]),
            "samples": int(len(accepted)), "spin_rpm": rpm_mean,
            "spin_axis_camera": axis.tolist(), "axis_angle_p90_deg": angle_p90,
            "rpm_coefficient_of_variation": rpm_cv,
            "quality_median": float(np.median(q)),
            "disagreement_median": float(np.median(d)),
            "internally_consistent": consistent,
        })
    return summaries
