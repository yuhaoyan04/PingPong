from __future__ import annotations

from typing import Optional, Tuple
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

from .models import SpinEstimate, Track2D


@njit(cache=True)
def _irls_kernel(A, b, dimensions, threshold, minimum_features):
    feature_count = len(b) // dimensions
    ata = A.T @ A + np.eye(3) * 1e-12
    omega = np.linalg.solve(ata, A.T @ b)
    error = np.empty(feature_count, np.float64)
    for _ in range(3):
        for i in range(feature_count):
            total = 0.0
            for d in range(dimensions):
                row = i * dimensions + d
                value = A[row, 0] * omega[0] + A[row, 1] * omega[1] + A[row, 2] * omega[2] - b[row]
                total += value * value
            error[i] = np.sqrt(total)
        median = np.median(error)
        scale = 1.4826 * np.median(np.abs(error - median)) + 1e-9
        cutoff = max(2.5 * scale, 0.15 * threshold)
        weighted_ata = np.eye(3) * 1e-12
        weighted_atb = np.zeros(3, np.float64)
        for i in range(feature_count):
            weight = 1.0 / (1.0 + (error[i] / cutoff) ** 2)
            for d in range(dimensions):
                row = i * dimensions + d
                for j in range(3):
                    weighted_atb[j] += weight * A[row, j] * b[row]
                    for k in range(3):
                        weighted_ata[j, k] += weight * A[row, j] * A[row, k]
        omega = np.linalg.solve(weighted_ata, weighted_atb)
    for i in range(feature_count):
        total = 0.0
        for d in range(dimensions):
            row = i * dimensions + d
            value = A[row, 0] * omega[0] + A[row, 1] * omega[1] + A[row, 2] * omega[2] - b[row]
            total += value * value
        error[i] = np.sqrt(total)
    median = np.median(error)
    mad = np.median(np.abs(error - median)) + 1e-9
    gate = max(threshold, median + 2.5 * 1.4826 * mad)
    inliers = error <= gate
    count = np.sum(inliers)
    if count < minimum_features:
        return omega, inliers, np.inf, np.inf
    final_ata = np.eye(3) * 1e-12
    final_atb = np.zeros(3, np.float64)
    for i in range(feature_count):
        if not inliers[i]:
            continue
        for d in range(dimensions):
            row = i * dimensions + d
            for j in range(3):
                final_atb[j] += A[row, j] * b[row]
                for k in range(3):
                    final_ata[j, k] += A[row, j] * A[row, k]
    omega = np.linalg.solve(final_ata, final_atb)
    squared = 0.0
    for i in range(feature_count):
        if not inliers[i]:
            continue
        for d in range(dimensions):
            row = i * dimensions + d
            value = A[row, 0] * omega[0] + A[row, 1] * omega[1] + A[row, 2] * omega[2] - b[row]
            squared += value * value
    residual = np.sqrt(squared / max(count * dimensions, 1))
    condition = np.sqrt(np.linalg.cond(final_ata))
    return omega, inliers, residual, condition


def solve_sphere_angular_velocity(
    xy0: np.ndarray, xy1: np.ndarray, dt: float, minimum_features: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve omega in camera coordinates from normalized visible-sphere flow.

    Camera coordinates use +x right, +y down and +z away from the camera, so
    points on the visible hemisphere have negative relative z.
    """
    xy0 = np.asarray(xy0, np.float64); xy1 = np.asarray(xy1, np.float64)
    z = -np.sqrt(np.maximum(1.0 - np.sum(xy0 * xy0, axis=1), 1e-6))
    velocity = (xy1 - xy0) / max(float(dt), 1e-9)
    A = np.zeros((2 * len(xy0), 3), dtype=np.float64)
    b = velocity.reshape(-1)
    A[0::2, 1] = z;  A[0::2, 2] = -xy0[:, 1]
    A[1::2, 0] = -z; A[1::2, 2] = xy0[:, 0]
    return _robust_linear_solve(A, b, dt, minimum_features, 2)


def _robust_linear_solve(A, b, dt, minimum_features, dimensions):
    feature_count = len(b) // dimensions
    # IRLS gives bounded influence to mismatched event edges with a fixed number
    # of 3x3 solves. The old exhaustive pair search required hundreds of SVDs
    # for every 0.5 ms state and dominated the runtime.
    threshold = 0.035 / max(float(dt), 1e-9)
    if NUMBA_AVAILABLE:
        return _irls_kernel(A, b, dimensions, threshold, minimum_features)
    def normal_solve(row_weights=None):
        if row_weights is None:
            ata, atb = A.T @ A, A.T @ b
        else:
            weighted_a = A * row_weights[:, None]
            ata, atb = weighted_a.T @ weighted_a, weighted_a.T @ (b * row_weights)
        ridge = np.eye(3) * max(float(np.trace(ata)) * 1e-12, 1e-12)
        return np.linalg.solve(ata + ridge, atb)

    omega = normal_solve()
    weights = np.ones(feature_count, np.float64)
    for _ in range(3):
        error = np.linalg.norm((A @ omega - b).reshape(-1, dimensions), axis=1)
        median = float(np.median(error))
        scale = 1.4826 * float(np.median(np.abs(error - median))) + 1e-9
        cutoff = max(2.5 * scale, 0.15 * threshold)
        weights = 1.0 / (1.0 + (error / cutoff) ** 2)
        row_weights = np.sqrt(np.repeat(weights, dimensions))
        updated = normal_solve(row_weights)
        if np.linalg.norm(updated - omega) <= 1e-7 * max(np.linalg.norm(omega), 1.0):
            omega = updated
            break
        omega = updated
    error = np.linalg.norm((A @ omega - b).reshape(-1, dimensions), axis=1)
    median = float(np.median(error)); mad = float(np.median(np.abs(error - median))) + 1e-9
    inliers = error <= np.maximum(threshold, median + 2.5 * 1.4826 * mad)

    # A bounded deterministic fallback handles cases where the initial fit was
    # dominated by outliers while preserving a hard real-time cost ceiling.
    if np.count_nonzero(inliers) < minimum_features:
        all_pairs = list(combinations(range(feature_count), 2))
        if len(all_pairs) > 16:
            indices = np.linspace(0, len(all_pairs) - 1, 16, dtype=int)
            all_pairs = [all_pairs[i] for i in indices]
        best_key = (-1, -float("inf")); best = omega; best_inliers = inliers
        for pair in all_pairs:
            mask = np.zeros(feature_count, bool); mask[list(pair)] = True
            rows = np.repeat(mask, dimensions)
            sample_a, sample_b = A[rows], b[rows]
            candidate = np.linalg.solve(
                sample_a.T @ sample_a + np.eye(3) * 1e-12, sample_a.T @ sample_b)
            candidate_error = np.linalg.norm((A @ candidate - b).reshape(-1, dimensions), axis=1)
            consensus = candidate_error <= threshold
            key = (int(np.count_nonzero(consensus)),
                   -float(np.median(candidate_error[consensus])) if np.any(consensus) else -float("inf"))
            if key > best_key:
                best_key, best, best_inliers = key, candidate, consensus
        omega, inliers = best, best_inliers
    rows = np.repeat(inliers, dimensions)
    if np.count_nonzero(inliers) >= minimum_features:
        accepted_a, accepted_b = A[rows], b[rows]
        omega = np.linalg.solve(
            accepted_a.T @ accepted_a + np.eye(3) * 1e-12,
            accepted_a.T @ accepted_b)
        residual = float(np.sqrt(np.mean((A[rows] @ omega - b[rows]) ** 2)))
        condition = float(np.linalg.cond(A[rows]))
    else:
        residual, condition = float("inf"), float("inf")
    return omega, inliers, residual, condition


def image_points_to_unit_sphere(points_px, center_px, radius_px, camera_matrix):
    points = np.asarray(points_px, np.float64)
    center = np.asarray(center_px, np.float64)
    K = np.asarray(camera_matrix, np.float64)
    fx, fy = K[0, 0], K[1, 1]
    focal = 0.5 * (fx + fy)
    rho = max(float(radius_px) / focal, 1e-6)
    qx = (points[:, 0] - center[0]) / (fx * rho)
    qy = (points[:, 1] - center[1]) / (fy * rho)
    a = (center[0] - K[0, 2]) / fx
    b = (center[1] - K[1, 2]) / fy
    kx, ky = rho * qx + a, rho * qy + b
    qa = kx * kx + ky * ky + 1.0
    qb = 2.0 * (qx * kx + qy * ky)
    qc = qx * qx + qy * qy - 1.0
    discriminant = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
    sz = (-qb - np.sqrt(discriminant)) / (2.0 * qa)
    sx, sy = qx + kx * sz, qy + ky * sz
    sphere = np.column_stack((sx, sy, sz))
    sphere /= np.maximum(np.linalg.norm(sphere, axis=1, keepdims=True), 1e-9)
    return sphere


def solve_perspective_sphere_angular_velocity(
    points0_px, points1_px, center0_px, center1_px, radius0_px, radius1_px,
    camera_matrix, dt, minimum_features=4,
):
    s0 = image_points_to_unit_sphere(points0_px, center0_px, radius0_px, camera_matrix)
    s1 = image_points_to_unit_sphere(points1_px, center1_px, radius1_px, camera_matrix)
    velocity = (s1 - s0) / max(float(dt), 1e-9)
    A = np.zeros((3 * len(s0), 3), np.float64)
    A[0::3, 1] = s0[:, 2]; A[0::3, 2] = -s0[:, 1]
    A[1::3, 0] = -s0[:, 2]; A[1::3, 2] = s0[:, 0]
    A[2::3, 0] = s0[:, 1]; A[2::3, 1] = -s0[:, 0]
    return _robust_linear_solve(A, velocity.reshape(-1), dt, minimum_features, 3)


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rotation mapping unit vectors in source to target."""
    covariance = np.asarray(target, np.float64).T @ np.asarray(source, np.float64)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as an axis-angle vector, stable near zero and pi."""
    vector, _ = cv2.Rodrigues(np.asarray(rotation, np.float64))
    return vector.reshape(3)


def solve_rigid_sphere_rotation(
    sphere0: np.ndarray,
    sphere1: np.ndarray,
    dt: float,
    minimum_features: int = 4,
    angular_threshold_rad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a finite 3-D rotation to matched surface directions.

    Unlike the differential optical-flow equations, this remains valid when a
    fast ball rotates by a substantial angle between event surfaces.  A fixed,
    deterministic hypothesis budget bounds runtime and rejects LK mismatches.
    """
    s0 = np.asarray(sphere0, np.float64)
    s1 = np.asarray(sphere1, np.float64)
    if s0.shape != s1.shape or s0.ndim != 2 or s0.shape[1] != 3:
        raise ValueError("sphere0 and sphere1 must both have shape (N, 3)")
    count = len(s0)
    empty = np.zeros(count, dtype=bool)
    if count < minimum_features or dt <= 0:
        return np.zeros(3), empty, float("inf"), float("inf")
    s0 = s0 / np.maximum(np.linalg.norm(s0, axis=1, keepdims=True), 1e-12)
    s1 = s1 / np.maximum(np.linalg.norm(s1, axis=1, keepdims=True), 1e-12)

    sample_size = min(3, count)
    hypotheses = list(combinations(range(count), sample_size))
    if len(hypotheses) > 64:
        selected = np.linspace(0, len(hypotheses) - 1, 64, dtype=int)
        hypotheses = [hypotheses[i] for i in selected]
    best_key = (-1, -float("inf"))
    best_inliers = empty
    threshold = float(angular_threshold_rad)
    for sample in hypotheses:
        rotation = _kabsch_rotation(s0[list(sample)], s1[list(sample)])
        predicted = (rotation @ s0.T).T
        residual = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1.0, 1.0))
        inliers = residual <= threshold
        n_inliers = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if n_inliers else float("inf")
        key = (n_inliers, -median)
        if key > best_key:
            best_key, best_inliers = key, inliers
    if np.count_nonzero(best_inliers) < minimum_features:
        return np.zeros(3), best_inliers, float("inf"), float("inf")

    rotation = _kabsch_rotation(s0[best_inliers], s1[best_inliers])
    predicted = (rotation @ s0.T).T
    residual = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1.0, 1.0))
    # One refinement allows points just outside a minimal-sample hypothesis to
    # re-enter without an unbounded iterative optimizer.
    median = float(np.median(residual[best_inliers]))
    mad = float(np.median(np.abs(residual[best_inliers] - median))) + 1e-9
    refined = residual <= max(threshold, median + 2.5 * 1.4826 * mad)
    if np.count_nonzero(refined) >= minimum_features:
        best_inliers = refined
        rotation = _kabsch_rotation(s0[best_inliers], s1[best_inliers])
        predicted = (rotation @ s0.T).T
        residual = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1.0, 1.0))
    accepted = residual[best_inliers]
    rms = float(np.sqrt(np.mean(accepted * accepted)))
    # Geometry is ill conditioned if visible vectors span almost one line.
    singular = np.linalg.svd(s0[best_inliers], compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return _rotation_vector(rotation) / float(dt), best_inliers, rms / float(dt), condition


def solve_perspective_sphere_rotation(
    points0_px, points1_px, center0_px, center1_px, radius0_px, radius1_px,
    camera_matrix, dt, minimum_features=4, angular_threshold_rad=0.05,
):
    s0 = image_points_to_unit_sphere(points0_px, center0_px, radius0_px, camera_matrix)
    s1 = image_points_to_unit_sphere(points1_px, center1_px, radius1_px, camera_matrix)
    return solve_rigid_sphere_rotation(
        s0, s1, dt, minimum_features, angular_threshold_rad)


def _rotate_vectors(vectors: np.ndarray, omega: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation with one shared angular velocity."""
    vectors = np.asarray(vectors, np.float64)
    omega = np.asarray(omega, np.float64).reshape(3)
    dt = np.asarray(dt, np.float64).reshape(-1)
    magnitude = float(np.linalg.norm(omega))
    if magnitude < 1e-12:
        return vectors.copy()
    axis = omega / magnitude
    angle = magnitude * dt
    cosine = np.cos(angle)[:, None]; sine = np.sin(angle)[:, None]
    return (vectors * cosine + np.cross(axis, vectors) * sine +
            np.outer(np.sum(vectors * axis, axis=1), axis) * (1.0 - cosine))


@njit(cache=True, fastmath=True, nogil=True)
def _score_omega_candidates_kernel(s0, s1, dt, omegas, cosine_threshold):
    counts = np.zeros(len(omegas), np.int32)
    mean_cosine = np.full(len(omegas), -1.0, np.float64)
    for h in range(len(omegas)):
        wx, wy, wz = omegas[h, 0], omegas[h, 1], omegas[h, 2]
        magnitude = np.sqrt(wx * wx + wy * wy + wz * wz)
        if magnitude < 1e-12:
            continue
        ax, ay, az = wx / magnitude, wy / magnitude, wz / magnitude
        total = 0.0; count = 0
        for i in range(len(s0)):
            vx, vy, vz = s0[i, 0], s0[i, 1], s0[i, 2]
            angle = magnitude * dt[i]
            co, si = np.cos(angle), np.sin(angle)
            dot_axis = vx * ax + vy * ay + vz * az
            cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
            px = vx * co + cx * si + dot_axis * ax * (1.0 - co)
            py = vy * co + cy * si + dot_axis * ay * (1.0 - co)
            pz = vz * co + cz * si + dot_axis * az * (1.0 - co)
            value = px * s1[i, 0] + py * s1[i, 1] + pz * s1[i, 2]
            if value >= cosine_threshold:
                count += 1; total += value
        counts[h] = count
        if count:
            mean_cosine[h] = total / count
    return counts, mean_cosine


def solve_constant_sphere_rotation(
    sphere0: np.ndarray,
    sphere1: np.ndarray,
    dt_s: np.ndarray,
    minimum_correspondences: int = 20,
    angular_threshold_rad: float = 0.035,
    hypotheses: int = 1200,
    maximum_rpm: float = 12000.0,
    seed: int = 97,
    ransac_evaluation_limit: int = 0,
    finalist_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pool many sparse stereo/time correspondences into one constant spin."""
    s0 = np.asarray(sphere0, np.float64).reshape(-1, 3)
    s1 = np.asarray(sphere1, np.float64).reshape(-1, 3)
    dt = np.asarray(dt_s, np.float64).reshape(-1)
    if len(s0) != len(s1) or len(s0) != len(dt):
        raise ValueError("sphere vectors and dt_s must have matching lengths")
    finite = (np.isfinite(s0).all(axis=1) & np.isfinite(s1).all(axis=1) &
              np.isfinite(dt) & (dt > 0))
    s0, s1, dt = s0[finite], s1[finite], dt[finite]
    count = len(s0); empty = np.zeros(count, bool)
    if count < minimum_correspondences:
        return np.zeros(3), empty, float("inf"), float("inf")
    s0 /= np.maximum(np.linalg.norm(s0, axis=1, keepdims=True), 1e-12)
    s1 /= np.maximum(np.linalg.norm(s1, axis=1, keepdims=True), 1e-12)
    A = np.zeros((count, 3, 3), np.float64)
    A[:, 0, 1] = s0[:, 2]; A[:, 0, 2] = -s0[:, 1]
    A[:, 1, 0] = -s0[:, 2]; A[:, 1, 2] = s0[:, 0]
    A[:, 2, 0] = s0[:, 1]; A[:, 2, 1] = -s0[:, 0]
    velocity = (s1 - s0) / dt[:, None]
    maximum_rad_s = maximum_rpm * 2.0 * np.pi / 60.0
    rng = np.random.default_rng(seed)
    if count > ransac_evaluation_limit > 0:
        evaluation = np.sort(rng.choice(count, ransac_evaluation_limit, replace=False))
    else:
        evaluation = np.arange(count)
    hypotheses_omega = []
    cosine_threshold = float(np.cos(angular_threshold_rad))
    sample_size = min(2, count)
    for _ in range(hypotheses):
        sample = rng.choice(count, sample_size, replace=False)
        if sample_size == 2 and abs(float(np.dot(s0[sample[0]], s0[sample[1]]))) > .97:
            continue
        omega, *_ = np.linalg.lstsq(A[sample].reshape(-1, 3),
                                    velocity[sample].reshape(-1), rcond=None)
        if not np.isfinite(omega).all() or np.linalg.norm(omega) > maximum_rad_s:
            continue
        hypotheses_omega.append(omega)
    if not hypotheses_omega:
        return np.zeros(3), empty, float("inf"), float("inf")
    hypotheses_omega = np.asarray(hypotheses_omega, np.float64)
    counts, mean_cosine = _score_omega_candidates_kernel(
        s0[evaluation], s1[evaluation], dt[evaluation], hypotheses_omega, cosine_threshold)
    order = np.lexsort((mean_cosine, counts))
    finalist_indices = order[-min(finalist_count, len(order)):]
    best_key = (-1, -float("inf")); best_omega = None; best_inliers = empty
    for index in finalist_indices:
        candidate = hypotheses_omega[index]
        predicted = _rotate_vectors(s0, candidate, dt)
        residual = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1.0, 1.0))
        inliers = residual <= angular_threshold_rad
        n = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if n else float("inf")
        key = (n, -median)
        if key > best_key:
            best_key, best_omega, best_inliers = key, candidate, inliers
    if best_omega is None or np.count_nonzero(best_inliers) < minimum_correspondences:
        return np.zeros(3), best_inliers, float("inf"), float("inf")

    active = best_inliers.copy()
    omega = best_omega
    for _ in range(2):
        def residual_function(value):
            return (_rotate_vectors(s0[active], value, dt[active]) - s1[active]).reshape(-1)
        fit = least_squares(residual_function, omega, loss="soft_l1",
                            f_scale=max(angular_threshold_rad * .5, 1e-4), max_nfev=80)
        omega = fit.x
        predicted = _rotate_vectors(s0, omega, dt)
        angular = np.arccos(np.clip(np.sum(predicted * s1, axis=1), -1.0, 1.0))
        active = angular <= angular_threshold_rad
        if np.count_nonzero(active) < minimum_correspondences:
            break
    if np.linalg.norm(omega) > maximum_rad_s or np.count_nonzero(active) < minimum_correspondences:
        return np.zeros(3), active, float("inf"), float("inf")
    angular = np.arccos(np.clip(np.sum(_rotate_vectors(s0, omega, dt) * s1, axis=1), -1, 1))
    rms = float(np.sqrt(np.mean(angular[active] ** 2)))
    design = A[active].reshape(-1, 3)
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return omega, active, rms, condition


class ImagePlaneSpinEstimator:
    """Fit 3D angular velocity from marker constellations, with LK fallback."""

    def __init__(self, config: dict, camera_matrix: Optional[np.ndarray] = None):
        self.cfg = config
        self.camera_matrix = camera_matrix
        self.previous_image: Optional[np.ndarray] = None
        self.previous_track: Optional[Track2D] = None
        self.previous_dots: Optional[np.ndarray] = None

    def _stabilized_patch(self, image: np.ndarray, track: Track2D) -> np.ndarray:
        size = int(self.cfg.get("normalized_patch_size", 128))
        patch_radius = 0.5 * size
        scale = max(float(track.radius), 1.0) / patch_radius
        M = np.array([[scale, 0.0, float(track.x) - scale * patch_radius],
                      [0.0, scale, float(track.y) - scale * patch_radius]], np.float32)
        return cv2.warpAffine(image, M, (size, size),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _patch_to_image(self, points: np.ndarray, track: Track2D) -> np.ndarray:
        center = 0.5 * int(self.cfg.get("normalized_patch_size", 128))
        return ((np.asarray(points, np.float64) - center) *
                (float(track.radius) / center) + np.array([track.x, track.y]))

    def _dot_centers(self, patch: np.ndarray) -> np.ndarray:
        """Detect marker rings while masking the ball silhouette."""
        if self.cfg.get("dot_detector", "hough") == "blobs":
            return self._blob_centers(patch)
        size = patch.shape[0]
        masked = patch.copy()
        yy, xx = np.ogrid[:size, :size]
        masked[(xx - size / 2) ** 2 + (yy - size / 2) ** 2 > (0.80 * size / 2) ** 2] = 0
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(masked, (3, 3), 0.7), cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=float(self.cfg.get("dot_min_distance_px", 7)),
            param1=float(self.cfg.get("dot_canny_threshold", 35)),
            param2=float(self.cfg.get("dot_vote_threshold", 6)),
            minRadius=int(self.cfg.get("dot_min_radius_px", 2)),
            maxRadius=int(self.cfg.get("dot_max_radius_px", 14)))
        if circles is None:
            return np.empty((0, 3), np.float64)
        proposals = circles[0].astype(np.float64)
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        scored = []
        for cx, cy, radius in proposals:
            samples = []
            for delta in (-1.0, 0.0, 1.0):
                x = np.rint(cx + (radius + delta) * np.cos(angles)).astype(int)
                y = np.rint(cy + (radius + delta) * np.sin(angles)).astype(int)
                valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
                values = np.zeros(len(angles), np.uint8)
                values[valid] = masked[y[valid], x[valid]]
                samples.append(values)
            ring = np.max(samples, axis=0)
            coverage = np.count_nonzero(ring >= 16) / len(ring)
            strength = float(np.mean(ring)) / 255.0
            score = 0.8 * coverage + 0.2 * strength
            if score >= float(self.cfg.get("dot_minimum_ring_score", 0.16)):
                scored.append((score, cx, cy, radius))
        kept = []
        separation = float(self.cfg.get("dot_nms_distance_px", 5.0))
        for score, cx, cy, radius in sorted(scored, reverse=True):
            if all(np.hypot(cx - old[0], cy - old[1]) >= separation for old in kept):
                kept.append((cx, cy, radius))
            if len(kept) >= int(self.cfg.get("maximum_dots", 16)):
                break
        return np.asarray(kept, np.float64).reshape(-1, 3)

    def _blob_centers(self, patch: np.ndarray) -> np.ndarray:
        """Detect compact event regions made by moving printed ball markers."""
        size = patch.shape[0]
        threshold = int(self.cfg.get("dot_blob_threshold", 20))
        binary = (patch >= threshold).astype(np.uint8) * 255
        yy, xx = np.ogrid[:size, :size]
        mask_radius = float(self.cfg.get("dot_mask_radius_fraction", 0.72)) * size / 2
        binary[(xx - size / 2) ** 2 + (yy - size / 2) ** 2 > mask_radius ** 2] = 0
        kernel_size = int(self.cfg.get("dot_blob_close_px", 3))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        minimum_area = int(self.cfg.get("dot_blob_min_area_px", 5))
        maximum_area = int(self.cfg.get("dot_blob_max_area_px", 350))
        maximum_aspect = float(self.cfg.get("dot_blob_max_aspect_ratio", 5.0))
        candidates = []
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            aspect = max(width, height) / max(min(width, height), 1)
            if not minimum_area <= area <= maximum_area or aspect > maximum_aspect:
                continue
            cx, cy = centroids[index]
            radius = np.sqrt(float(area) / np.pi)
            candidates.append((float(area), cx, cy, radius))
        candidates.sort(reverse=True)
        maximum = int(self.cfg.get("maximum_dots", 16))
        return np.asarray([(cx, cy, radius) for _, cx, cy, radius in candidates[:maximum]],
                          np.float64).reshape(-1, 3)

    def _match_dots(self, previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidates = []
        for i, old in enumerate(previous):
            for j, new in enumerate(current):
                ratio = max(old[2], new[2]) / max(min(old[2], new[2]), 1e-6)
                distance = float(np.linalg.norm(old[:2] - new[:2]))
                if (distance <= float(self.cfg.get("dot_max_motion_px", 28)) and
                        ratio <= float(self.cfg.get("dot_max_radius_ratio", 2.0))):
                    candidates.append((distance + 3.0 * abs(np.log(ratio)), i, j))
        used_old, used_new, pairs = set(), set(), []
        for _, i, j in sorted(candidates):
            if i not in used_old and j not in used_new:
                used_old.add(i); used_new.add(j); pairs.append((i, j))
        if not pairs:
            empty = np.empty((0, 2), np.float64)
            return empty, empty
        return (np.array([previous[i, :2] for i, _ in pairs]),
                np.array([current[j, :2] for _, j in pairs]))

    @staticmethod
    def _gray(surface: np.ndarray) -> np.ndarray:
        return np.clip(np.maximum(surface[0], surface[1]) * 255, 0, 255).astype(np.uint8)

    def _estimate_points(self, p0_patch: np.ndarray, p1_patch: np.ndarray,
                         previous_track: Track2D, track: Track2D, dt: float,
                         minimum_features: int) -> Optional[SpinEstimate]:
        p0_image = self._patch_to_image(p0_patch, previous_track)
        p1_image = self._patch_to_image(p1_patch, track)
        center0 = np.array([previous_track.x, previous_track.y])
        center1 = np.array([track.x, track.y])
        radius0, radius1 = max(previous_track.radius, 1.0), max(track.radius, 1.0)
        xy0 = (p0_image - center0) / radius0
        xy1 = (p1_image - center1) / radius1
        inside = ((np.sum(xy0 * xy0, axis=1) < 0.82 ** 2) &
                  (np.sum(xy1 * xy1, axis=1) < 0.90 ** 2))
        xy0, xy1 = xy0[inside], xy1[inside]
        if len(xy0) < minimum_features:
            return None
        finite_rotation = self.cfg.get("solver", "rigid_rotation") == "rigid_rotation"
        if self.camera_matrix is not None and finite_rotation:
            omega, inliers, residual, condition = solve_perspective_sphere_rotation(
                p0_image[inside], p1_image[inside], center0, center1, radius0, radius1,
                self.camera_matrix, dt, minimum_features,
                float(self.cfg.get("rotation_inlier_threshold_rad", 0.05)))
        elif self.camera_matrix is not None:
            omega, inliers, residual, condition = solve_perspective_sphere_angular_velocity(
                p0_image[inside], p1_image[inside], center0, center1, radius0, radius1,
                self.camera_matrix, dt, minimum_features)
        else:
            omega, inliers, residual, condition = solve_sphere_angular_velocity(
                xy0, xy1, dt, minimum_features)
        used = int(np.count_nonzero(inliers))
        motion = np.linalg.norm((xy1 - xy0) / dt, axis=1)
        relative_residual = residual / max(float(np.median(motion[inliers])) if used else 0.0, 1e-3)
        quality = float(np.exp(-2.0 * relative_residual) * min(used / 8.0, 1.0) *
                        min(30.0 / max(condition, 1.0), 1.0))
        rpm = float(np.linalg.norm(omega) * 60.0 / (2.0 * np.pi))
        if (rpm > self.cfg["max_rpm"] or not np.isfinite(condition) or condition > 100.0 or
                quality < float(self.cfg.get("minimum_quality", 0.15))):
            return None
        return SpinEstimate(track.timestamp_us, omega, rpm, used, quality)

    def _remember(self, image: np.ndarray, track: Track2D, dots: np.ndarray) -> None:
        self.previous_image, self.previous_track, self.previous_dots = image, track, dots

    def update(self, surface: np.ndarray, track: Optional[Track2D],
               normalized_patch: bool = False) -> Optional[SpinEstimate]:
        full_current = self._gray(surface)
        # Between expensive circle detections the 2D tracker supplies a predicted
        # center. Keeping the normalized patch alive enables a high-rate spin
        # channel without running Hough on every 0.5 ms step.
        if track is None or track.radius < self.cfg["minimum_radius_px"] or track.confidence <= 0:
            self.previous_image, self.previous_track, self.previous_dots = None, None, None
            return None
        current = full_current if normalized_patch else self._stabilized_patch(full_current, track)
        dots_enabled = (int(self.cfg.get("minimum_dot_features", 99)) <=
                        int(self.cfg.get("maximum_dots", 16)))
        current_dots = self._dot_centers(current) if dots_enabled else np.empty((0, 3), np.float64)
        if self.previous_image is None or self.previous_track is None:
            self._remember(current, track, current_dots); return None
        dt = (track.timestamp_us - self.previous_track.timestamp_us) * 1e-6
        if dt <= 0:
            return None
        if self.previous_dots is not None:
            dot0, dot1 = self._match_dots(self.previous_dots, current_dots)
            minimum_dots = int(self.cfg.get("minimum_dot_features", 3))
            if len(dot0) >= minimum_dots:
                estimate = self._estimate_points(dot0, dot1, self.previous_track, track, dt, minimum_dots)
                if estimate is not None:
                    self._remember(current, track, current_dots); return estimate

        mask = np.zeros_like(current); patch_center = current.shape[0] // 2
        cv2.circle(mask, (patch_center, patch_center), round(patch_center * 0.82), 255, -1)
        points = cv2.goodFeaturesToTrack(
            self.previous_image, maxCorners=int(self.cfg["maximum_features"]),
            qualityLevel=0.02, minDistance=2, mask=mask, blockSize=3)
        if points is None or len(points) < self.cfg["minimum_features"]:
            self._remember(current, track, current_dots); return None
        lk_window = int(self.cfg.get("lk_window_px", 15))
        lk_levels = int(self.cfg.get("lk_pyramid_levels", 2))
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_image, current, points, None,
            winSize=(lk_window, lk_window), maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02))
        if next_points is None or status is None or error is None:
            self._remember(current, track, current_dots); return None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current, self.previous_image, next_points, None,
            winSize=(lk_window, lk_window), maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        if back_points is None or back_status is None:
            self._remember(current, track, current_dots); return None
        fb_error = np.linalg.norm(back_points.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
        good = (status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool) &
                (fb_error <= float(self.cfg.get("max_forward_backward_error_px", 1.5))) &
                (error.reshape(-1) <= float(self.cfg.get("max_lk_error", 30.0))))
        p0, p1 = points.reshape(-1, 2)[good], next_points.reshape(-1, 2)[good]
        estimate = None
        if len(p0) >= self.cfg["minimum_features"]:
            estimate = self._estimate_points(
                p0, p1, self.previous_track, track, dt, int(self.cfg["minimum_features"]))
        self._remember(current, track, current_dots)
        return estimate


def fuse_optical_axis_spin(
    omega_left: np.ndarray,
    omega_right: np.ndarray,
    right_to_left_rotation: np.ndarray,
    left_quality: float = 1.0,
    right_quality: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Rotate both full-vector estimates into the left camera frame and fuse."""
    wl = np.asarray(omega_left, dtype=np.float64)
    wr = np.asarray(right_to_left_rotation, dtype=np.float64) @ np.asarray(omega_right, dtype=np.float64)
    weights = np.maximum([left_quality, right_quality], 1e-3)
    fused = (weights[0] * wl + weights[1] * wr) / weights.sum()
    disagreement = np.linalg.norm(wl - wr) / max(np.linalg.norm(fused), 1.0)
    quality = float(np.sqrt(left_quality * right_quality) * np.exp(-disagreement))
    return fused, quality
