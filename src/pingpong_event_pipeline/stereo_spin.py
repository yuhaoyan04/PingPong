from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from itertools import combinations, permutations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .stereo import StereoCalibration
from .spin import _kabsch_rotation, _rotation_vector


@dataclass
class SphereMarkerSet:
    directions: np.ndarray
    points_m: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray
    reprojection_error_px: np.ndarray
    shell_error_m: np.ndarray


@dataclass
class TemplatePoseSpinEstimate:
    omega_rad_s: np.ndarray
    axis: np.ndarray
    rpm: float | None
    pose_frames: int
    candidate_pairs: int
    inlier_pairs: int
    span_s: float
    residual_rms_rad: float
    axis_resultant: float
    accepted: bool
    reason: str


@dataclass
class LabeledSphereTemplate:
    marker_ids: np.ndarray
    directions: np.ndarray
    observation_counts: np.ndarray
    frame_indices: np.ndarray
    timestamps_us: np.ndarray
    poses: np.ndarray
    frame_residuals: np.ndarray
    accepted: bool
    reason: str


@dataclass
class StreamingTemplateSpinResult:
    timestamp_us: int
    pose_valid: bool
    pose_mode: str
    template_inliers: int
    pose_residual_chord: float
    window_pose_frames: int
    spin: TemplatePoseSpinEstimate | None

    @property
    def rpm(self) -> float | None:
        return None if self.spin is None or not self.spin.accepted else self.spin.rpm


def _sphere_triplet_signature(points: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray([
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[1] - points[2]),
    ], np.float64))


def audit_sphere_template_symmetry(
    template: np.ndarray,
    signature_tolerance: float = .035,
    match_chord_distance: float = .06,
    minimum_nontrivial_rotation_rad: float = .08,
    maximum_hypotheses: int = 30000,
    maximum_nontrivial_match_fraction: float = .70,
) -> dict:
    """Search for non-identity rotations that map a marker template onto itself."""
    points = np.asarray(template, np.float64).reshape(-1, 3)
    if len(points) < 4:
        return {
            "passed": False, "markers": len(points), "hypotheses_tested": 0,
            "maximum_nontrivial_match_fraction": None,
            "reason": "fewer than four template markers",
        }
    triplets = np.asarray(list(combinations(range(len(points)), 3)), np.int32)
    signatures = np.asarray([_sphere_triplet_signature(points[item])
                             for item in triplets], np.float64)
    tree = cKDTree(signatures)
    nearest = tree.query(signatures, k=2)[0][:, 1]
    tested = 0
    best_count = 0
    best_residual = float("inf")
    best_angle = None
    stop = False
    for source_number, source_indices in enumerate(triplets):
        neighbors = tree.query_ball_point(
            signatures[source_number], signature_tolerance, p=np.inf)
        for target_number in neighbors:
            target_indices = triplets[target_number]
            for ordering in permutations(range(3)):
                tested += 1
                target = target_indices[list(ordering)]
                rotation = _kabsch_rotation(points[source_indices], points[target])
                angle = float(np.linalg.norm(_rotation_vector(rotation)))
                if angle < minimum_nontrivial_rotation_rad:
                    if tested >= maximum_hypotheses:
                        stop = True
                        break
                    continue
                transformed = (rotation @ points.T).T
                distances = np.linalg.norm(
                    transformed[:, None, :] - points[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(distances)
                valid = distances[rows, cols] <= match_chord_distance
                count = int(np.count_nonzero(valid))
                residual = (float(np.sqrt(np.mean(
                    distances[rows[valid], cols[valid]] ** 2)))
                            if count else float("inf"))
                if count > best_count or (count == best_count and residual < best_residual):
                    best_count, best_residual, best_angle = count, residual, angle
                if count == len(points) and residual <= .25 * match_chord_distance:
                    stop = True
                    break
                if tested >= maximum_hypotheses:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    fraction = best_count / len(points)
    passed = fraction < maximum_nontrivial_match_fraction
    return {
        "passed": bool(passed),
        "markers": len(points),
        "triplets": len(triplets),
        "hypotheses_tested": tested,
        "signature_nearest_neighbor_median": float(np.median(nearest)),
        "signature_nearest_neighbor_p10": float(np.percentile(nearest, 10)),
        "signature_collision_fraction_within_tolerance": float(np.mean(
            nearest <= signature_tolerance)),
        "maximum_nontrivial_match_count": best_count,
        "maximum_nontrivial_match_fraction": fraction,
        "best_nontrivial_residual_chord": best_residual if np.isfinite(best_residual) else None,
        "best_nontrivial_rotation_rad": best_angle,
        "acceptance_threshold": maximum_nontrivial_match_fraction,
        "reason": ("no high-coverage rotational self-symmetry found" if passed else
                   "template has a high-coverage non-identity rotational self-match"),
    }


class SphereTemplateMatcher:
    """Precomputed rotation-invariant triplet matcher for a calibrated template."""

    def __init__(self, template: np.ndarray, signature_tolerance: float = .035,
                 inlier_chord_distance: float = .08,
                 minimum_markers: int = 3, maximum_hypotheses: int = 5000,
                 early_stop_unmatched_observations: int = 1):
        self.template = np.asarray(template, np.float64).reshape(-1, 3)
        self.signature_tolerance = float(signature_tolerance)
        self.inlier_chord_distance = float(inlier_chord_distance)
        self.minimum_markers = int(minimum_markers)
        self.maximum_hypotheses = int(maximum_hypotheses)
        self.early_stop_unmatched_observations = int(early_stop_unmatched_observations)
        self.triplets = np.asarray(
            list(combinations(range(len(self.template)), 3)), np.int32).reshape(-1, 3)
        self.signatures = np.asarray([
            _sphere_triplet_signature(self.template[indices])
            for indices in self.triplets
        ], np.float64).reshape(-1, 3)
        self.tree = cKDTree(self.signatures) if len(self.signatures) else None

    def rotation_hypotheses(self, observations: np.ndarray,
                            observation_keys: np.ndarray | None = None,
                            maximum_hypotheses: int | None = None,
                            maximum_template_triplets_per_observation: int = 8):
        """Yield triplet-derived poses without prematurely choosing one.

        ``observation_keys`` can hold the left/right candidate indices used to
        triangulate each observation.  Triplets that reuse a candidate in
        either image cannot represent three physical markers and are skipped.
        The caller can then score each pose jointly in both camera images.
        """
        observed = np.asarray(observations, np.float64).reshape(-1, 3)
        if self.tree is None or len(observed) < 3:
            return
        keys = None if observation_keys is None else np.asarray(
            observation_keys, np.int32).reshape(-1, 2)
        limit = (self.maximum_hypotheses if maximum_hypotheses is None
                 else int(maximum_hypotheses))
        tested = 0
        for observation_tuple in combinations(range(len(observed)), 3):
            observation_indices = np.asarray(observation_tuple, np.int32)
            if keys is not None:
                selected = keys[observation_indices]
                if len(np.unique(selected[:, 0])) < 3 or len(np.unique(selected[:, 1])) < 3:
                    continue
            signature = _sphere_triplet_signature(observed[observation_indices])
            candidates = self.tree.query_ball_point(
                signature, self.signature_tolerance, p=np.inf)
            candidates = sorted(candidates, key=lambda candidate: float(
                np.linalg.norm(self.signatures[candidate] - signature)))
            for candidate in candidates[:maximum_template_triplets_per_observation]:
                template_indices = self.triplets[candidate]
                for ordering in permutations(range(3)):
                    if tested >= limit:
                        return
                    tested += 1
                    ordered = observation_indices[list(ordering)]
                    yield _kabsch_rotation(
                        self.template[template_indices], observed[ordered])

    def match(self, observations: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        observed = np.asarray(observations, np.float64).reshape(-1, 3)
        empty = np.empty(0, np.int32)
        if (self.tree is None or len(observed) < self.minimum_markers
                or len(self.template) < self.minimum_markers):
            return np.eye(3), empty, empty.copy(), float("inf"), 0
        best_key = (-1, -float("inf")); best = None; tested = 0
        target_count = max(
            self.minimum_markers,
            min(len(self.template), len(observed)) - self.early_stop_unmatched_observations)
        early_stop = False
        for observation_tuple in combinations(range(len(observed)), 3):
            observation_indices = np.asarray(observation_tuple, np.int32)
            signature = _sphere_triplet_signature(observed[observation_indices])
            candidate_triplets = self.tree.query_ball_point(
                signature, self.signature_tolerance, p=np.inf)
            for candidate in candidate_triplets:
                template_indices = self.triplets[candidate]
                for ordering in permutations(range(3)):
                    tested += 1
                    if tested > self.maximum_hypotheses:
                        break
                    ordered_observations = observation_indices[list(ordering)]
                    rotation = _kabsch_rotation(
                        self.template[template_indices], observed[ordered_observations])
                    transformed = (rotation @ self.template.T).T
                    distance = np.linalg.norm(
                        transformed[:, None, :] - observed[None, :, :], axis=2)
                    rows, cols = linear_sum_assignment(distance)
                    valid = distance[rows, cols] <= self.inlier_chord_distance
                    count = int(np.count_nonzero(valid))
                    residual = (float(np.mean(distance[rows[valid], cols[valid]]))
                                if count else float("inf"))
                    key = (count, -residual)
                    if key > best_key:
                        best_key, best = key, (rows[valid], cols[valid])
                    if (count >= target_count
                            and residual <= .5 * self.inlier_chord_distance):
                        early_stop = True
                        break
                if early_stop:
                    break
                if tested > self.maximum_hypotheses:
                    break
            if early_stop:
                break
            if tested > self.maximum_hypotheses:
                break
        if best is None or len(best[0]) < self.minimum_markers:
            return np.eye(3), empty, empty.copy(), float("inf"), tested
        rows, cols = best
        rotation = _kabsch_rotation(self.template[rows], observed[cols])
        transformed = (rotation @ self.template[rows].T).T
        residual = float(np.sqrt(np.mean(np.sum(
            (transformed - observed[cols]) ** 2, axis=1))))
        return (rotation, rows.astype(np.int32), cols.astype(np.int32),
                residual, tested)


class SphereTemplatePoseTracker:
    """Track a calibrated sphere template with global acquisition and local updates."""

    def __init__(self, template: np.ndarray, signature_tolerance: float = .035,
                 inlier_chord_distance: float = .08, minimum_markers: int = 4,
                 maximum_hypotheses: int = 5000, maximum_gap_us: int = 5000,
                 velocity_smoothing: float = .25,
                 early_stop_unmatched_observations: int = 1):
        self.matcher = SphereTemplateMatcher(
            template, signature_tolerance, inlier_chord_distance,
            minimum_markers, maximum_hypotheses,
            early_stop_unmatched_observations)
        self.template = self.matcher.template
        self.threshold = float(inlier_chord_distance)
        self.minimum_markers = int(minimum_markers)
        self.maximum_gap_us = int(maximum_gap_us)
        self.velocity_smoothing = float(velocity_smoothing)
        self.pose = None
        self.timestamp_us = None
        self.omega = None

    def reset(self) -> None:
        self.pose = None
        self.timestamp_us = None
        self.omega = None

    def _local_match(self, observed: np.ndarray, prediction: np.ndarray):
        empty = np.empty(0, np.int32)
        rotation = prediction
        rows = cols = empty
        for _ in range(3):
            transformed = (rotation @ self.template.T).T
            distance = np.linalg.norm(
                transformed[:, None, :] - observed[None, :, :], axis=2)
            candidate_rows, candidate_cols = linear_sum_assignment(distance)
            valid = distance[candidate_rows, candidate_cols] <= self.threshold
            rows, cols = candidate_rows[valid], candidate_cols[valid]
            if len(rows) < self.minimum_markers:
                return np.eye(3), empty, empty.copy(), float("inf")
            rotation = _kabsch_rotation(self.template[rows], observed[cols])
        transformed = (rotation @ self.template[rows].T).T
        residual = float(np.sqrt(np.mean(np.sum(
            (transformed - observed[cols]) ** 2, axis=1))))
        return rotation, rows.astype(np.int32), cols.astype(np.int32), residual

    def _commit_pose(self, timestamp: int, rotation: np.ndarray) -> None:
        if self.pose is not None and self.timestamp_us is not None:
            dt = (timestamp - self.timestamp_us) * 1e-6
            if 0 < dt <= self.maximum_gap_us * 1e-6:
                instantaneous = _rotation_vector(rotation @ self.pose.T) / dt
                if self.omega is None:
                    self.omega = instantaneous
                else:
                    alpha = np.clip(self.velocity_smoothing, 0.0, 1.0)
                    self.omega = (1 - alpha) * self.omega + alpha * instantaneous
        self.pose = rotation
        self.timestamp_us = timestamp

    def update_corresponded(self, timestamp_us: int, template_indices: np.ndarray,
                           observations: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str, int]:
        rows = np.asarray(template_indices, np.int32).reshape(-1)
        observed = np.asarray(observations, np.float64).reshape(-1, 3)
        empty = np.empty(0, np.int32)
        if len(rows) < self.minimum_markers or len(rows) != len(observed):
            return np.eye(3), empty, empty.copy(), float("inf"), "guided", 0
        rotation = _kabsch_rotation(self.template[rows], observed)
        transformed = (rotation @ self.template[rows].T).T
        residual = float(np.sqrt(np.mean(np.sum((transformed - observed) ** 2, axis=1))))
        if residual > self.threshold:
            return rotation, empty, empty.copy(), residual, "guided", 0
        self._commit_pose(int(timestamp_us), rotation)
        return rotation, rows, np.arange(len(rows), dtype=np.int32), residual, "guided", 0

    def update(self, timestamp_us: int, observations: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str, int]:
        observed = np.asarray(observations, np.float64).reshape(-1, 3)
        timestamp = int(timestamp_us)
        use_local = False
        if self.pose is not None and self.omega is not None and self.timestamp_us is not None:
            gap_us = timestamp - self.timestamp_us
            if 0 < gap_us <= self.maximum_gap_us:
                delta, _ = cv2.Rodrigues(self.omega * (gap_us * 1e-6))
                prediction = delta @ self.pose
                rotation, rows, cols, residual = self._local_match(observed, prediction)
                use_local = len(rows) >= self.minimum_markers
        if not use_local:
            rotation, rows, cols, residual, tested = self.matcher.match(observed)
            mode = "global"
        else:
            tested = 0
            mode = "local"
        if len(rows) < self.minimum_markers:
            return rotation, rows, cols, residual, mode, tested
        self._commit_pose(timestamp, rotation)
        return rotation, rows, cols, residual, mode, tested


class CalibratedTemplateSpinStream:
    """Causal rolling-window template pose and 3-D spin estimator."""

    def __init__(self, template: np.ndarray, template_validated: bool,
                 window_us: int = 30000, reset_gap_us: int = 6000,
                 minimum_markers: int = 4, signature_tolerance: float = .035,
                 inlier_chord_distance: float = .06,
                 spin_update_interval_us: int = 0):
        self.template_validated = bool(template_validated)
        self.window_us = int(window_us)
        self.reset_gap_us = int(reset_gap_us)
        self.minimum_markers = int(minimum_markers)
        self.inlier_chord_distance = float(inlier_chord_distance)
        self.spin_update_interval_us = int(spin_update_interval_us)
        self.tracker = SphereTemplatePoseTracker(
            template, signature_tolerance=signature_tolerance,
            inlier_chord_distance=inlier_chord_distance,
            minimum_markers=minimum_markers, maximum_gap_us=reset_gap_us,
            early_stop_unmatched_observations=2)
        self.timestamps = deque()
        self.poses = deque()
        self.last_timestamp = None
        self.last_spin_timestamp = None
        self.last_spin = None

    def reset(self) -> None:
        self.tracker.reset()
        self.timestamps.clear()
        self.poses.clear()
        self.last_timestamp = None
        self.last_spin_timestamp = None
        self.last_spin = None

    def _current_spin(self, timestamp: int):
        if len(self.poses) < 7:
            return None
        if (self.last_spin is None or self.last_spin_timestamp is None
                or self.spin_update_interval_us <= 0
                or timestamp - self.last_spin_timestamp >= self.spin_update_interval_us):
            self.last_spin = estimate_spin_from_template_poses(
                np.asarray(self.timestamps, np.int64), np.asarray(self.poses, np.float64),
                minimum_pairs=6, minimum_span_s=.006,
                maximum_gap_us=self.reset_gap_us,
                inlier_threshold_rad=.04, maximum_rms_rad=.025,
                minimum_axis_resultant=.85)
            self.last_spin_timestamp = timestamp
        return self.last_spin

    def update(self, timestamp_us: int, directions: np.ndarray) -> StreamingTemplateSpinResult:
        timestamp = int(timestamp_us)
        if (self.last_timestamp is not None
                and (timestamp <= self.last_timestamp
                     or timestamp - self.last_timestamp > self.reset_gap_us)):
            self.reset()
        self.last_timestamp = timestamp
        if not self.template_validated:
            return StreamingTemplateSpinResult(
                timestamp, False, "disabled_unvalidated_template", 0,
                float("inf"), 0, None)
        pose, rows, _, residual, mode, _ = self.tracker.update(timestamp, directions)
        valid = len(rows) >= self.minimum_markers and residual <= self.inlier_chord_distance
        if valid:
            self.timestamps.append(timestamp)
            self.poses.append(pose)
        while self.timestamps and timestamp - self.timestamps[0] > self.window_us:
            self.timestamps.popleft(); self.poses.popleft()
        spin = self._current_spin(timestamp)
        return StreamingTemplateSpinResult(
            timestamp, valid, mode, len(rows), residual, len(self.poses), spin)

    def update_corresponded(self, timestamp_us: int, template_indices: np.ndarray,
                           directions: np.ndarray) -> StreamingTemplateSpinResult:
        timestamp = int(timestamp_us)
        if (self.last_timestamp is not None
                and (timestamp <= self.last_timestamp
                     or timestamp - self.last_timestamp > self.reset_gap_us)):
            self.reset()
        self.last_timestamp = timestamp
        if not self.template_validated:
            return StreamingTemplateSpinResult(
                timestamp, False, "disabled_unvalidated_template", 0,
                float("inf"), 0, None)
        pose, rows, _, residual, mode, _ = self.tracker.update_corresponded(
            timestamp, template_indices, directions)
        valid = len(rows) >= self.minimum_markers and residual <= self.inlier_chord_distance
        if valid:
            self.timestamps.append(timestamp); self.poses.append(pose)
        while self.timestamps and timestamp - self.timestamps[0] > self.window_us:
            self.timestamps.popleft(); self.poses.popleft()
        spin = self._current_spin(timestamp)
        return StreamingTemplateSpinResult(
            timestamp, valid, mode, len(rows), residual, len(self.poses), spin)


def estimate_spin_from_template_poses(
    timestamps_us: np.ndarray,
    rotations: np.ndarray,
    minimum_pairs: int = 6,
    minimum_span_s: float = .006,
    maximum_gap_us: int = 3000,
    maximum_pair_rotation_rad: float = 2.8,
    inlier_threshold_rad: float = .04,
    maximum_rms_rad: float = .025,
    minimum_axis_resultant: float = .85,
) -> TemplatePoseSpinEstimate:
    """Robustly estimate one camera-frame angular velocity from absolute poses.

    Each pose maps the calibrated body-fixed marker template into the camera
    coordinate system. The relative rotation is therefore ``R1 @ R0.T``.
    RPM is emitted only when temporal support, residual, and axis-consistency
    gates all pass.
    """
    timestamps = np.asarray(timestamps_us, np.int64).reshape(-1)
    poses = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    if len(timestamps) != len(poses):
        raise ValueError("timestamps and rotations must have equal length")
    empty = np.full(3, np.nan, np.float64)
    if len(poses) < 2:
        return TemplatePoseSpinEstimate(empty, empty.copy(), None, len(poses), 0, 0,
                                        0.0, float("inf"), 0.0, False,
                                        "fewer than two valid template poses")
    order = np.argsort(timestamps, kind="stable")
    timestamps, poses = timestamps[order], poses[order]
    rotvecs = []
    durations = []
    for index in range(1, len(poses)):
        gap_us = int(timestamps[index] - timestamps[index - 1])
        if gap_us <= 0 or gap_us > maximum_gap_us:
            continue
        relative = poses[index] @ poses[index - 1].T
        vector = _rotation_vector(relative)
        angle = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or angle > maximum_pair_rotation_rad:
            continue
        rotvecs.append(vector)
        durations.append(gap_us * 1e-6)
    if not rotvecs:
        return TemplatePoseSpinEstimate(empty, empty.copy(), None, len(poses), 0, 0,
                                        0.0, float("inf"), 0.0, False,
                                        "no valid adjacent pose pairs")
    vectors = np.asarray(rotvecs, np.float64)
    dt = np.asarray(durations, np.float64)
    omega = np.median(vectors / dt[:, None], axis=0)
    threshold = max(float(inlier_threshold_rad), 1e-6)
    for _ in range(12):
        residual = np.linalg.norm(vectors - dt[:, None] * omega, axis=1)
        weights = np.minimum(1.0, threshold / np.maximum(residual, 1e-12))
        denominator = float(np.sum(weights * dt * dt))
        if denominator <= 1e-18:
            break
        updated = np.sum((weights * dt)[:, None] * vectors, axis=0) / denominator
        if np.linalg.norm(updated - omega) <= 1e-8 * max(1.0, np.linalg.norm(omega)):
            omega = updated
            break
        omega = updated
    residual = np.linalg.norm(vectors - dt[:, None] * omega, axis=1)
    inliers = residual <= threshold
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count:
        denominator = float(np.sum(dt[inliers] ** 2))
        omega = np.sum(dt[inliers, None] * vectors[inliers], axis=0) / max(denominator, 1e-18)
        residual = np.linalg.norm(vectors - dt[:, None] * omega, axis=1)
        inliers = residual <= threshold
        inlier_count = int(np.count_nonzero(inliers))
    rms = (float(np.sqrt(np.mean(residual[inliers] ** 2)))
           if inlier_count else float("inf"))
    magnitude = float(np.linalg.norm(omega))
    axis = omega / magnitude if magnitude > 1e-12 else empty.copy()
    useful = inliers & (np.linalg.norm(vectors, axis=1) > 1e-6)
    if np.any(useful) and np.isfinite(axis).all():
        pair_axes = vectors[useful] / np.linalg.norm(vectors[useful], axis=1, keepdims=True)
        signs = np.sign(pair_axes @ axis)
        signs[signs == 0] = 1
        axis_resultant = float(np.linalg.norm(np.mean(pair_axes * signs[:, None], axis=0)))
    else:
        axis_resultant = 0.0
    span_s = float((timestamps[-1] - timestamps[0]) * 1e-6)
    checks = [
        (inlier_count >= minimum_pairs,
         f"requires at least {minimum_pairs} inlier pose pairs"),
        (span_s >= minimum_span_s,
         f"requires at least {minimum_span_s:g} s pose span"),
        (magnitude > 1e-6, "angular speed is indistinguishable from zero"),
        (rms <= maximum_rms_rad,
         f"rotation residual RMS exceeds {maximum_rms_rad:g} rad"),
        (axis_resultant >= minimum_axis_resultant,
         f"axis resultant is below {minimum_axis_resultant:g}"),
    ]
    failed = [reason for passed, reason in checks if not passed]
    accepted = not failed
    rpm = magnitude * 60.0 / (2.0 * np.pi) if accepted else None
    return TemplatePoseSpinEstimate(
        omega, axis, rpm, len(poses), len(vectors), inlier_count, span_s,
        rms, axis_resultant, accepted,
        "accepted calibrated-template pose sequence" if accepted else "; ".join(failed))


def build_labeled_sphere_template(
    labeled_frames: list[dict],
    minimum_shared_markers: int = 3,
    maximum_frame_residual: float = .04,
    minimum_template_markers: int = 8,
    minimum_processed_frames: int = 10,
    minimum_observations_per_marker: int = 2,
) -> LabeledSphereTemplate:
    """Fuse human-identified 3-D sphere directions into one body-fixed template.

    Each frame is ``{"timestamp_us": int, "directions": {physical_id: vec3}}``.
    The first usable frame defines the arbitrary body coordinate system. Later
    frames require at least three already-known IDs to estimate their pose; new
    IDs are then rotated into the body frame and added to the template.
    """
    empty_i = np.empty(0, np.int64)
    empty_v = np.empty((0, 3), np.float64)
    normalized = []
    for source_index, frame in enumerate(labeled_frames):
        directions = {}
        for identifier, vector in frame.get("directions", {}).items():
            value = np.asarray(vector, np.float64).reshape(3)
            norm = float(np.linalg.norm(value))
            if np.isfinite(value).all() and norm > 1e-12:
                directions[int(identifier)] = value / norm
        if len(directions) >= minimum_shared_markers:
            normalized.append((source_index, int(frame["timestamp_us"]), directions))
    if not normalized:
        return LabeledSphereTemplate(
            empty_i, empty_v, empty_i.copy(), empty_i.copy(), empty_i.copy(),
            np.empty((0, 3, 3)), np.empty(0), False,
            "no labeled frame has enough valid shared 3-D marker directions")
    seed_position = int(np.argmax([len(item[2]) for item in normalized]))
    seed = normalized.pop(seed_position)
    samples = {identifier: [direction.copy()] for identifier, direction in seed[2].items()}
    template = {identifier: direction.copy() for identifier, direction in seed[2].items()}
    processed = [(seed[0], seed[1], np.eye(3), 0.0)]
    remaining = normalized
    while remaining:
        progressed = False
        deferred = []
        for source_index, timestamp, observed in remaining:
            overlap = sorted(set(template) & set(observed))
            if len(overlap) < minimum_shared_markers:
                deferred.append((source_index, timestamp, observed))
                continue
            reference = np.asarray([template[item] for item in overlap])
            current = np.asarray([observed[item] for item in overlap])
            pose = _kabsch_rotation(reference, current)
            predicted = (pose @ reference.T).T
            residual = float(np.sqrt(np.mean(np.sum((predicted - current) ** 2, axis=1))))
            if residual > maximum_frame_residual:
                deferred.append((source_index, timestamp, observed))
                continue
            for identifier, direction in observed.items():
                body_direction = pose.T @ direction
                body_direction /= max(np.linalg.norm(body_direction), 1e-12)
                samples.setdefault(identifier, []).append(body_direction)
                mean = np.mean(samples[identifier], axis=0)
                template[identifier] = mean / max(np.linalg.norm(mean), 1e-12)
            processed.append((source_index, timestamp, pose, residual))
            progressed = True
        if not progressed:
            break
        remaining = deferred
    identifiers = np.asarray(sorted(template), np.int64)
    directions = np.asarray([template[item] for item in identifiers], np.float64)
    counts = np.asarray([len(samples[item]) for item in identifiers], np.int64)
    processed.sort(key=lambda item: item[1])
    frame_indices = np.asarray([item[0] for item in processed], np.int64)
    timestamps = np.asarray([item[1] for item in processed], np.int64)
    poses = np.asarray([item[2] for item in processed], np.float64)
    residuals = np.asarray([item[3] for item in processed], np.float64)
    supported = int(np.count_nonzero(counts >= minimum_observations_per_marker))
    checks = [
        (len(processed) >= minimum_processed_frames,
         f"requires at least {minimum_processed_frames} connected labeled frames"),
        (supported >= minimum_template_markers,
         f"requires at least {minimum_template_markers} markers with repeated observations"),
        (len(remaining) == 0, "some labeled frames are disconnected or geometrically inconsistent"),
        (float(np.max(residuals, initial=0.0)) <= maximum_frame_residual,
         f"frame residual exceeds {maximum_frame_residual:g} chord distance"),
    ]
    failed = [reason for passed, reason in checks if not passed]
    return LabeledSphereTemplate(
        identifiers, directions, counts, frame_indices, timestamps, poses, residuals,
        not failed, "accepted human-identified sphere template" if not failed
        else "; ".join(failed))


def triangulate_sphere_markers(
    left_points_px: np.ndarray,
    right_points_px: np.ndarray,
    calibration: StereoCalibration,
    center_m: np.ndarray,
    radius_m: float = 0.020,
    maximum_reprojection_px: float = 2.5,
    maximum_shell_error_m: float = 0.006,
) -> SphereMarkerSet:
    """Stereo-match marker candidates using reprojection and sphere radius."""
    left = np.asarray(left_points_px, np.float64).reshape(-1, 2)
    right = np.asarray(right_points_px, np.float64).reshape(-1, 2)
    center = np.asarray(center_m, np.float64).reshape(3)
    shape = (len(left), len(right))
    costs = np.full(shape, 1e6, np.float64)
    points = np.zeros(shape + (3,), np.float64)
    reprojection = np.full(shape, np.inf, np.float64)
    shell_error = np.full(shape, np.inf, np.float64)
    for i, lp in enumerate(left):
        for j, rp in enumerate(right):
            point, error = calibration.triangulate(lp, rp)
            shell = abs(float(np.linalg.norm(point - center)) - float(radius_m))
            if (np.isfinite(point).all() and error <= maximum_reprojection_px and
                    shell <= maximum_shell_error_m):
                costs[i, j] = error / maximum_reprojection_px + shell / maximum_shell_error_m
                points[i, j] = point; reprojection[i, j] = error; shell_error[i, j] = shell
    if not len(left) or not len(right):
        empty_i = np.empty(0, np.int32); empty_v = np.empty((0, 3), np.float64)
        return SphereMarkerSet(empty_v, empty_v.copy(), empty_i, empty_i.copy(),
                               np.empty(0), np.empty(0))
    rows, cols = linear_sum_assignment(costs)
    valid = costs[rows, cols] < 1e5
    rows, cols = rows[valid], cols[valid]
    accepted_points = points[rows, cols]
    directions = accepted_points - center
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return SphereMarkerSet(directions, accepted_points, rows.astype(np.int32), cols.astype(np.int32),
                           reprojection[rows, cols], shell_error[rows, cols])


def triangulate_sphere_marker_hypotheses(
    left_points_px: np.ndarray,
    right_points_px: np.ndarray,
    calibration: StereoCalibration,
    center_m: np.ndarray,
    radius_m: float = .020,
    maximum_reprojection_px: float = 2.5,
    maximum_shell_error_m: float = .006,
    maximum_pairs_per_left: int = 3,
    maximum_pairs_per_right: int = 3,
) -> SphereMarkerSet:
    """Keep several plausible stereo pairs for robust template acquisition.

    Unlike ``triangulate_sphere_markers``, this deliberately does not commit to
    one anonymous Hungarian pairing before the physical template is known.
    """
    left = np.asarray(left_points_px, np.float64).reshape(-1, 2)
    right = np.asarray(right_points_px, np.float64).reshape(-1, 2)
    center = np.asarray(center_m, np.float64).reshape(3)
    candidates = []
    for i, lp in enumerate(left):
        local = []
        for j, rp in enumerate(right):
            point, reprojection = calibration.triangulate(lp, rp)
            shell = abs(float(np.linalg.norm(point - center)) - float(radius_m))
            if (np.isfinite(point).all() and reprojection <= maximum_reprojection_px
                    and shell <= maximum_shell_error_m):
                cost = (reprojection / max(maximum_reprojection_px, 1e-12)
                        + shell / max(maximum_shell_error_m, 1e-12))
                local.append((cost, i, j, point, reprojection, shell))
        candidates.extend(sorted(local, key=lambda item: item[0])[:maximum_pairs_per_left])
    accepted = []
    right_usage = {}
    for item in sorted(candidates, key=lambda value: value[0]):
        right_index = item[2]
        if right_usage.get(right_index, 0) >= maximum_pairs_per_right:
            continue
        accepted.append(item)
        right_usage[right_index] = right_usage.get(right_index, 0) + 1
    if not accepted:
        empty_i = np.empty(0, np.int32); empty_v = np.empty((0, 3), np.float64)
        return SphereMarkerSet(empty_v, empty_v.copy(), empty_i, empty_i.copy(),
                               np.empty(0), np.empty(0))
    points = np.asarray([item[3] for item in accepted], np.float64)
    directions = points - center
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return SphereMarkerSet(
        directions, points,
        np.asarray([item[1] for item in accepted], np.int32),
        np.asarray([item[2] for item in accepted], np.int32),
        np.asarray([item[4] for item in accepted], np.float64),
        np.asarray([item[5] for item in accepted], np.float64))


def match_sphere_marker_sets(
    previous: np.ndarray,
    current: np.ndarray,
    maximum_chord_distance: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Match unordered 3-D surface directions for a small temporal step."""
    a = np.asarray(previous, np.float64).reshape(-1, 3)
    b = np.asarray(current, np.float64).reshape(-1, 3)
    if not len(a) or not len(b):
        empty = np.empty((0, 3), np.float64)
        return empty, empty
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distance)
    valid = distance[rows, cols] <= maximum_chord_distance
    return a[rows[valid]], b[cols[valid]]


def estimate_unordered_sphere_rotation(
    previous: np.ndarray,
    current: np.ndarray,
    maximum_rotation_rad: float = 0.25,
    inlier_chord_distance: float = 0.10,
    minimum_markers: int = 3,
    hypotheses: int = 1200,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate rotation between unordered marker sets with bounded RANSAC."""
    a = np.asarray(previous, np.float64).reshape(-1, 3)
    b = np.asarray(current, np.float64).reshape(-1, 3)
    if len(a) < minimum_markers or len(b) < minimum_markers:
        empty = np.empty((0, 3), np.float64)
        return np.eye(3), empty, empty, float("inf")
    rng = np.random.default_rng(seed)
    best_key = (-1, -float("inf")); best = None
    for _ in range(hypotheses):
        ia = rng.choice(len(a), 3, replace=False)
        ib = rng.choice(len(b), 3, replace=False)
        rotation = _kabsch_rotation(a[ia], b[ib])
        if np.linalg.norm(_rotation_vector(rotation)) > maximum_rotation_rad:
            continue
        transformed = (rotation @ a.T).T
        distance = np.linalg.norm(transformed[:, None, :] - b[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distance)
        valid = distance[rows, cols] <= inlier_chord_distance
        count = int(np.count_nonzero(valid))
        residual = float(np.mean(distance[rows[valid], cols[valid]])) if count else float("inf")
        key = (count, -residual)
        if key > best_key:
            best_key, best = key, (rows[valid], cols[valid])
    if best is None or len(best[0]) < minimum_markers:
        empty = np.empty((0, 3), np.float64)
        return np.eye(3), empty, empty, float("inf")
    rows, cols = best
    rotation = _kabsch_rotation(a[rows], b[cols])
    transformed = (rotation @ a[rows].T).T
    residual = float(np.sqrt(np.mean(np.sum((transformed - b[cols]) ** 2, axis=1))))
    return rotation, a[rows], b[cols], residual


def estimate_sphere_template_pose(
    template: np.ndarray,
    observations: np.ndarray,
    inlier_chord_distance: float = 0.10,
    minimum_markers: int = 3,
    hypotheses: int = 4000,
    seed: int = 71,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Align unordered observations to a fixed sphere template.

    Indices expose proposed physical correspondences. An automatically
    bootstrapped template remains diagnostic until physically calibrated.
    """
    reference = np.asarray(template, np.float64).reshape(-1, 3)
    observed = np.asarray(observations, np.float64).reshape(-1, 3)
    if len(reference) < minimum_markers or len(observed) < minimum_markers:
        empty = np.empty(0, np.int32)
        return np.eye(3), empty, empty.copy(), float("inf")
    rng = np.random.default_rng(seed)
    best_key = (-1, -float("inf")); best = None
    for _ in range(hypotheses):
        reference_indices = rng.choice(len(reference), 3, replace=False)
        observed_indices = rng.choice(len(observed), 3, replace=False)
        rotation = _kabsch_rotation(
            reference[reference_indices], observed[observed_indices])
        transformed = (rotation @ reference.T).T
        distance = np.linalg.norm(
            transformed[:, None, :] - observed[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distance)
        valid = distance[rows, cols] <= inlier_chord_distance
        count = int(np.count_nonzero(valid))
        residual = (float(np.mean(distance[rows[valid], cols[valid]]))
                    if count else float("inf"))
        key = (count, -residual)
        if key > best_key:
            best_key, best = key, (rows[valid], cols[valid])
    if best is None or len(best[0]) < minimum_markers:
        empty = np.empty(0, np.int32)
        return np.eye(3), empty, empty.copy(), float("inf")
    rows, cols = best
    rotation = _kabsch_rotation(reference[rows], observed[cols])
    transformed = (rotation @ reference[rows].T).T
    residual = float(np.sqrt(np.mean(np.sum(
        (transformed - observed[cols]) ** 2, axis=1))))
    return rotation, rows.astype(np.int32), cols.astype(np.int32), residual
