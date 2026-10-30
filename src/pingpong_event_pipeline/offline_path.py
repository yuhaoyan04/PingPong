from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .models import Detection2D
from .stereo import StereoCalibration


@dataclass
class SmoothImagePath:
    timestamps_us: np.ndarray
    centers_px: np.ndarray
    radii_px: np.ndarray
    selected: np.ndarray
    residual_px: np.ndarray

    def evaluate(self, timestamps_us: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(timestamps_us, np.float64)
        center = np.column_stack((
            np.interp(query, self.timestamps_us, self.centers_px[:, 0]),
            np.interp(query, self.timestamps_us, self.centers_px[:, 1]),
        ))
        radius = np.interp(query, self.timestamps_us, self.radii_px)
        dt = np.gradient(self.timestamps_us) * 1e-6
        velocity_samples = np.column_stack((
            np.gradient(self.centers_px[:, 0]) / dt,
            np.gradient(self.centers_px[:, 1]) / dt,
        ))
        velocity = np.column_stack((
            np.interp(query, self.timestamps_us, velocity_samples[:, 0]),
            np.interp(query, self.timestamps_us, velocity_samples[:, 1]),
        ))
        return center, radius, velocity


@dataclass
class SmoothStereoPath:
    timestamps_us: np.ndarray
    points_m: np.ndarray
    left_centers_px: np.ndarray
    right_centers_px: np.ndarray
    left_radii_px: np.ndarray
    right_radii_px: np.ndarray
    selected: np.ndarray
    residual_m: np.ndarray
    pair_counts: np.ndarray


def _project_stereo_points(points_m: np.ndarray, calibration: StereoCalibration):
    points = np.asarray(points_m, np.float64).reshape(-1, 3)
    left, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                calibration.K1, calibration.D1)
    rvec, _ = cv2.Rodrigues(calibration.R)
    right, _ = cv2.projectPoints(points, rvec, calibration.T.reshape(3),
                                 calibration.K2, calibration.D2)
    return left.reshape(-1, 2), right.reshape(-1, 2)


def fit_smooth_stereo_candidate_path(
    timestamps_us: Sequence[int],
    left_candidates: Sequence[Sequence[Detection2D]],
    right_candidates: Sequence[Sequence[Detection2D]],
    calibration: StereoCalibration,
    ball_radius_m: float = 0.020,
    maximum_candidates: int = 18,
    maximum_pairs: int = 80,
    max_epipolar_error_px: float = 4.0,
    max_reprojection_error_px: float = 3.0,
    maximum_radius_log_error: float = 0.8,
    minimum_depth_m: float = 0.3,
    maximum_depth_m: float = 3.0,
    hypotheses: int = 8000,
    inlier_threshold_m: float = 0.045,
    seed: int = 23,
) -> SmoothStereoPath:
    """Fit one physical 3-D flight through jointly valid stereo circle pairs."""
    times = np.asarray(timestamps_us, np.float64)
    if not (len(times) == len(left_candidates) == len(right_candidates)) or len(times) < 3:
        raise ValueError("timestamps and stereo candidates need at least three matching bins")
    pair_bins = []
    for left_values, right_values in zip(left_candidates, right_candidates):
        left_ranked = sorted(left_values, key=lambda value: value.score, reverse=True)[:maximum_candidates]
        right_ranked = sorted(right_values, key=lambda value: value.score, reverse=True)[:maximum_candidates]
        pairs = []
        for left in left_ranked:
            for right in right_ranked:
                epi = calibration.symmetric_epipolar_error(left.point, right.point)
                if epi > max_epipolar_error_px:
                    continue
                point, reprojection = calibration.triangulate(left.point, right.point)
                right_point = calibration.R @ point + calibration.T.reshape(3)
                if (not np.isfinite(point).all() or reprojection > max_reprojection_error_px or
                        not minimum_depth_m <= point[2] <= maximum_depth_m or
                        not minimum_depth_m <= right_point[2] <= maximum_depth_m):
                    continue
                expected_left = calibration.K1[0, 0] * ball_radius_m / point[2]
                expected_right = calibration.K2[0, 0] * ball_radius_m / right_point[2]
                left_radius_error = abs(float(np.log(max(left.radius, 1e-6) / expected_left)))
                right_radius_error = abs(float(np.log(max(right.radius, 1e-6) / expected_right)))
                if (left_radius_error > maximum_radius_log_error or
                        right_radius_error > maximum_radius_log_error):
                    continue
                cost = (epi / max_epipolar_error_px + reprojection / max_reprojection_error_px +
                        left_radius_error + right_radius_error)
                score = float(left.score + right.score - 0.25 * cost)
                pairs.append((score, point, left.point, right.point))
        pairs.sort(key=lambda value: value[0], reverse=True)
        pair_bins.append(pairs[:maximum_pairs])
    usable = [index for index, pairs in enumerate(pair_bins) if pairs]
    if len(usable) < 3:
        raise ValueError("not enough bins contain physically valid stereo pairs")

    origin = float(np.mean(times)); scale = max(float(np.ptp(times)) * 0.5, 1.0)
    u = (times - origin) / scale
    design = np.column_stack((np.ones(len(u)), u, u * u))
    padded = np.full((len(times), maximum_pairs, 3), np.nan, np.float64)
    pair_scores = np.full((len(times), maximum_pairs), -np.inf, np.float64)
    for i, pairs in enumerate(pair_bins):
        for j, (score, point, _, _) in enumerate(pairs):
            padded[i, j] = point; pair_scores[i, j] = score

    thirds = [list(values) for values in np.array_split(np.asarray(usable), 3)]
    if any(not values for values in thirds):
        raise ValueError("stereo pairs do not span enough time")
    rng = np.random.default_rng(seed)
    best_key = (-1, -float("inf")); best_coefficients = None
    for _ in range(hypotheses):
        indices = [int(rng.choice(values)) for values in thirds]
        points = []
        for index in indices:
            values = pair_bins[index]
            scores = np.asarray([value[0] for value in values], np.float64)
            weights = np.exp(np.clip(scores - np.max(scores), -20, 0))
            points.append(values[int(rng.choice(len(values), p=weights / weights.sum()))][1])
        try:
            coefficients = np.linalg.solve(design[indices], np.asarray(points))
        except np.linalg.LinAlgError:
            continue
        velocity = ((coefficients[1][None, :] + 2 * u[:, None] * coefficients[2][None, :]) /
                    (scale * 1e-6))
        acceleration = 2 * coefficients[2] / (scale * 1e-6) ** 2
        if np.max(np.linalg.norm(velocity, axis=1)) > 30 or np.linalg.norm(acceleration) > 150:
            continue
        predicted = design @ coefficients
        distance = np.linalg.norm(padded - predicted[:, None, :], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        nearest = np.argmin(distance, axis=1)
        residual = distance[np.arange(len(times)), nearest]
        inliers = residual <= inlier_threshold_m
        count = int(np.count_nonzero(inliers))
        quality = np.sum(pair_scores[np.arange(len(times))[inliers], nearest[inliers]])
        key = (count, float(quality - 10 * np.sum(residual[inliers])))
        if key > best_key:
            best_key, best_coefficients = key, coefficients
    if best_coefficients is None:
        raise RuntimeError("could not fit a plausible joint stereo path")

    selected = np.full(len(times), -1, np.int32)
    coefficients = best_coefficients
    for _ in range(5):
        predicted = design @ coefficients
        rows = []; points = []
        for i in usable:
            values = np.asarray([value[1] for value in pair_bins[i]])
            j = int(np.argmin(np.linalg.norm(values - predicted[i], axis=1)))
            if np.linalg.norm(values[j] - predicted[i]) <= inlier_threshold_m:
                selected[i] = j; rows.append(i); points.append(values[j])
        if len(rows) < 3:
            break
        coefficients = np.linalg.lstsq(design[rows], np.asarray(points), rcond=None)[0]
    points_m = design @ coefficients
    residual = np.full(len(times), np.inf)
    for i in usable:
        values = np.asarray([value[1] for value in pair_bins[i]])
        j = int(np.argmin(np.linalg.norm(values - points_m[i], axis=1)))
        residual[i] = float(np.linalg.norm(values[j] - points_m[i]))
        selected[i] = j if residual[i] <= inlier_threshold_m else -1
    left_centers, right_centers = _project_stereo_points(points_m, calibration)
    right_points = (calibration.R @ points_m.T).T + calibration.T.reshape(1, 3)
    left_radii = calibration.K1[0, 0] * ball_radius_m / points_m[:, 2]
    right_radii = calibration.K2[0, 0] * ball_radius_m / right_points[:, 2]
    return SmoothStereoPath(
        times.astype(np.int64), points_m, left_centers, right_centers,
        left_radii, right_radii, selected, residual,
        np.asarray([len(values) for values in pair_bins], np.int32))


def fit_smooth_candidate_path(
    timestamps_us: Sequence[int],
    candidates: Sequence[Sequence[Detection2D]],
    maximum_candidates: int = 15,
    hypotheses: int = 6000,
    inlier_threshold_px: float = 28.0,
    minimum_span_us: int = 40_000,
    seed: int = 17,
) -> SmoothImagePath:
    """Link ambiguous circle candidates by fitting one smooth image trajectory."""
    times = np.asarray(timestamps_us, np.float64)
    if len(times) != len(candidates) or len(times) < 3:
        raise ValueError("timestamps and candidates need at least three matching bins")
    usable = [i for i, values in enumerate(candidates) if values]
    if len(usable) < 3:
        raise ValueError("not enough bins containing circle candidates")
    origin = float(np.mean(times))
    scale = max(float(np.ptp(times)) * 0.5, 1.0)
    u = (times - origin) / scale
    arrays = []
    for values in candidates:
        ranked = sorted(values, key=lambda item: item.score, reverse=True)[:maximum_candidates]
        arrays.append(np.array([[d.x, d.y, d.radius, d.score] for d in ranked], np.float64))
    padded_xy = np.full((len(times), maximum_candidates, 2), np.nan, np.float64)
    padded_score = np.zeros((len(times), maximum_candidates), np.float64)
    for i, values in enumerate(arrays):
        n = min(len(values), maximum_candidates)
        if n:
            padded_xy[i, :n] = values[:n, :2]
            padded_score[i, :n] = values[:n, 3]

    early = [i for i in usable if u[i] < -0.25]
    middle = [i for i in usable if abs(u[i]) <= 0.30]
    late = [i for i in usable if u[i] > 0.25]
    if not early or not middle or not late:
        thirds = np.array_split(np.asarray(usable), 3)
        early, middle, late = map(list, thirds)
    rng = np.random.default_rng(seed)
    best_key = (-1, -float("inf")); best_coefficients = None
    design = np.column_stack((np.ones(len(u)), u, u * u))
    for _ in range(hypotheses):
        frame_indices = [int(rng.choice(group)) for group in (early, middle, late)]
        if times[frame_indices[-1]] - times[frame_indices[0]] < minimum_span_us:
            continue
        points = []
        for index in frame_indices:
            values = arrays[index]
            # Larger circles are the full silhouette more often than small arcs.
            weights = np.maximum(values[:, 2], 1.0) * np.maximum(values[:, 3], 0.03)
            choice = int(rng.choice(len(values), p=weights / weights.sum()))
            points.append(values[choice, :2])
        sample_design = design[frame_indices]
        try:
            coefficients = np.linalg.solve(sample_design, np.asarray(points))
        except np.linalg.LinAlgError:
            continue
        predicted = design @ coefficients
        # Reject implausible paths before scoring all candidates.
        velocity = coefficients[1] / scale + 2.0 * u[:, None] * coefficients[2] / scale
        if np.max(np.linalg.norm(velocity, axis=1)) * 1e6 > 40_000:
            continue
        squared = np.sum((padded_xy - predicted[:, None, :]) ** 2, axis=2)
        squared[~np.isfinite(squared)] = np.inf
        nearest = np.argmin(squared, axis=1)
        distances = np.sqrt(squared[np.arange(len(times)), nearest])
        scores = padded_score[np.arange(len(times)), nearest]
        inliers = distances <= inlier_threshold_px
        count = int(np.count_nonzero(inliers))
        key = (count, float(np.sum(scores[inliers]) - 0.02 * np.sum(distances[inliers])))
        if key > best_key:
            best_key, best_coefficients = key, coefficients
    if best_coefficients is None:
        raise RuntimeError("could not fit a plausible image path")

    selected = np.full(len(times), -1, np.int32)
    coefficients = best_coefficients
    for _ in range(4):
        predicted = design @ coefficients
        chosen_points = []; chosen_rows = []
        for i in usable:
            delta = arrays[i][:, :2] - predicted[i]
            j = int(np.argmin(np.sum(delta * delta, axis=1)))
            if np.linalg.norm(delta[j]) <= inlier_threshold_px:
                selected[i] = j; chosen_rows.append(i); chosen_points.append(arrays[i][j, :2])
        if len(chosen_rows) < 3:
            break
        coefficients = np.linalg.lstsq(design[chosen_rows], np.asarray(chosen_points), rcond=None)[0]
    centers = design @ coefficients
    residual = np.full(len(times), np.inf)
    radii_t = []; radii_y = []
    for i in usable:
        delta = arrays[i][:, :2] - centers[i]
        j = int(np.argmin(np.sum(delta * delta, axis=1)))
        residual[i] = np.linalg.norm(delta[j])
        if residual[i] <= inlier_threshold_px:
            selected[i] = j; radii_t.append(u[i]); radii_y.append(arrays[i][j, 2])
        else:
            selected[i] = -1
    radius_coefficients = np.polyfit(radii_t, radii_y, min(2, len(radii_t) - 1))
    radii = np.polyval(radius_coefficients, u)
    return SmoothImagePath(times.astype(np.int64), centers, radii, selected, residual)
