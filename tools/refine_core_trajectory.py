"""Robustly smooth a short monocular ball trajectory for event compensation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def robust_polynomial(x: np.ndarray, y: np.ndarray, degree: int,
                      base_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vander(x, degree + 1, increasing=True)
    weight = np.ones(len(x), np.float64) if base_weight is None else np.asarray(base_weight, np.float64)
    weight = np.clip(weight, .05, 1.0)
    coefficients = np.zeros(degree + 1)
    for _ in range(10):
        weighted = matrix * np.sqrt(weight[:, None])
        coefficients = np.linalg.lstsq(weighted, y * np.sqrt(weight), rcond=None)[0]
        residual = y - matrix @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        huber = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
        new_weight = np.clip(weight * huber, .01, 1.0)
        if np.max(np.abs(new_weight - weight)) < 1e-4:
            break
        weight = new_weight
    return coefficients, weight


def evaluate(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vander(x, len(coefficients), increasing=True) @ coefficients


def derivative(coefficients: np.ndarray, x: np.ndarray, time_scale_s: float) -> np.ndarray:
    derived = np.arange(1, len(coefficients)) * coefficients[1:]
    if len(derived) == 0:
        return np.zeros_like(x)
    return evaluate(derived, x) / time_scale_s


def robust_local_polynomial(timestamps: np.ndarray, values: np.ndarray,
                            measured: np.ndarray, score_weight: np.ndarray,
                            degree: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a robust local polynomial at every timestamp.

    A single polynomial is too rigid for perspective-distorted image trajectories.
    This moving fit follows the physical path while suppressing detector jitter.
    """
    sample_indices = np.flatnonzero(measured)
    half = max(window // 2, degree + 1)
    predictions = np.empty(len(timestamps), np.float64)
    derivatives = np.empty(len(timestamps), np.float64)
    confidence = np.empty(len(timestamps), np.float64)
    for output_index, timestamp in enumerate(timestamps):
        nearest = int(np.searchsorted(timestamps[sample_indices], timestamp))
        start = max(0, nearest - half)
        stop = min(len(sample_indices), start + 2 * half + 1)
        start = max(0, stop - (2 * half + 1))
        local_indices = sample_indices[start:stop]
        delta_s = (timestamps[local_indices] - timestamp) * 1e-6
        span = max(float(np.max(np.abs(delta_s))), 1e-6)
        normalized = delta_s / span
        distance_weight = np.maximum(1.0 - np.abs(normalized) ** 3, 0.0) ** 3
        base_weight = np.clip(score_weight[local_indices] * distance_weight, .01, 1.0)
        local_degree = min(degree, len(local_indices) - 2)
        coeff, robust_weight = robust_polynomial(
            normalized, values[local_indices], local_degree, base_weight)
        predictions[output_index] = coeff[0]
        derivatives[output_index] = (coeff[1] / span) if len(coeff) > 1 else 0.0
        confidence[output_index] = float(np.mean(robust_weight))
    return predictions, derivatives, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="鲁棒短时球轨迹拟合")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--window", type=int, default=11,
                        help="odd local-fit window in samples (default: 11)")
    args = parser.parse_args()
    source = np.load(args.path)
    timestamps = source["timestamps_us"].astype(np.int64)
    centers = source["centers_px"].astype(np.float64)
    radii = source["radii_px"].astype(np.float64)
    scores = source["scores"].astype(np.float64)
    measured = scores > 0
    if np.count_nonzero(measured) < args.degree + 3:
        measured[:] = True
    score_weight = np.ones(len(scores), np.float64)
    positive_scores = scores[measured]
    score_weight[measured] = np.clip(
        positive_scores / max(float(np.percentile(positive_scores, 90)), 1e-6), .15, 1.0)
    score_weight[~measured] = .05
    fitted = np.empty((len(timestamps), 3), np.float64)
    velocities = np.empty((len(timestamps), 2), np.float64)
    final_weights = []
    observations = np.column_stack((centers, np.log(np.maximum(radii, 1.0))))
    for axis in range(3):
        prediction, velocity, weight = robust_local_polynomial(
            timestamps, observations[:, axis], measured, score_weight,
            args.degree, args.window)
        if axis < 2:
            fitted[:, axis] = prediction
            velocities[:, axis] = velocity
        else:
            fitted[:, axis] = np.exp(prediction)
        final_weights.append(weight)
    residual = np.linalg.norm(centers[measured] - fitted[measured, :2], axis=1)
    radius_relative = np.abs(radii[measured] - fitted[measured, 2]) / fitted[measured, 2]
    raw_acceleration = np.diff(centers, n=2, axis=0)
    refined_acceleration = np.diff(fitted[:, :2], n=2, axis=0)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, timestamps_us=timestamps, centers_px=fitted[:, :2],
             radii_px=fitted[:, 2], scores=scores, supports=source["supports"],
             velocity_px_s=velocities, original_centers_px=centers,
             original_radii_px=radii, fit_confidence=np.asarray(final_weights, np.float64))
    report = {
        "purpose": "robust_short_horizon_ball_trajectory_for_event_compensation",
        "source": str(Path(args.path).resolve()), "output": str(output.resolve()),
        "timestamps": len(timestamps), "measured_timestamps": int(np.count_nonzero(measured)),
        "model": "robust_local_polynomial", "polynomial_degree": args.degree,
        "local_window_samples": args.window,
        "center_fit_residual_px_median": float(np.median(residual)),
        "center_fit_residual_px_p90": float(np.percentile(residual, 90)),
        "radius_relative_residual_median": float(np.median(radius_relative)),
        "image_speed_px_s_median": float(np.median(np.linalg.norm(velocities, axis=1))),
        "raw_second_difference_px_median": float(np.median(np.linalg.norm(raw_acceleration, axis=1))) if len(raw_acceleration) else 0.0,
        "refined_second_difference_px_median": float(np.median(np.linalg.norm(refined_acceleration, axis=1))) if len(refined_acceleration) else 0.0,
        "warning": "This is a smooth image-space path, not stereo 3D ground truth.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
