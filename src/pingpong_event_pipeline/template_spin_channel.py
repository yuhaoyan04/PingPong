from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .events import EVENT_DTYPE, motion_compensated_ball_patch
from .models import Track2D
from .spin import ImagePlaneSpinEstimator
from .stereo import StereoCalibration
from .stereo_spin import CalibratedTemplateSpinStream, StreamingTemplateSpinResult


@dataclass
class StereoTemplateSpinChannelResult:
    timestamp_us: int
    master_candidates: int
    slave_candidates: int
    triangulated_markers: int
    stream: Optional[StreamingTemplateSpinResult]

    @property
    def rpm(self) -> Optional[float]:
        return None if self.stream is None else self.stream.rpm


def load_validated_template(template_path: str | Path, report_path: str | Path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    required = {
        "physical_template_geometry_accepted": report.get(
            "physical_template_geometry_accepted", False),
        "candidate_tracking_identity_accepted": report.get(
            "candidate_tracking_identity_accepted", False),
        "spin_solver_allowed": report.get("spin_solver_allowed", False),
    }
    failed = [name for name, value in required.items() if not value]
    if failed:
        raise ValueError("template report failed gates: " + ", ".join(failed))
    data = np.load(Path(template_path))
    directions = data["directions"].astype(np.float64)
    if len(directions) < 4 or not np.isfinite(directions).all():
        raise ValueError("validated template needs at least four finite directions")
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions, report


class StereoTemplateSpinChannel:
    """Causal event batches to triangulated markers and validated template spin."""

    def __init__(self, calibration_path: str | Path, template_path: str | Path,
                 template_report_path: str | Path, config: Optional[dict] = None):
        template, self.template_report = load_validated_template(
            template_path, template_report_path)
        self.calibration = StereoCalibration(calibration_path)
        defaults = {
            "window_us": 2000,
            "tau_us": 1000,
            "patch_size": 64,
            "ball_radius_m": .020,
            "maximum_reprojection_px": 3.0,
            "maximum_shell_error_m": .006,
            "normalized_patch_size": 64,
            "dot_detector": "blobs",
            "maximum_dots": 18,
            "dot_blob_threshold": 20,
            "dot_mask_radius_fraction": .72,
            "dot_blob_close_px": 3,
            "dot_blob_min_area_px": 5,
            "dot_blob_max_area_px": 350,
            "dot_blob_max_aspect_ratio": 5.0,
            "warmup_event_kernel": True,
        }
        self.config = {**defaults, **(config or {})}
        self.detectors = (
            ImagePlaneSpinEstimator(self.config, self.calibration.K1),
            ImagePlaneSpinEstimator(self.config, self.calibration.K2),
        )
        self.stream = CalibratedTemplateSpinStream(
            template, template_validated=True,
            window_us=int(self.config.get("pose_window_us", 30000)),
            reset_gap_us=int(self.config.get("reset_gap_us", 6000)),
            minimum_markers=int(self.config.get("minimum_markers", 4)),
            signature_tolerance=float(self.config.get("signature_tolerance", .035)),
            inlier_chord_distance=float(self.config.get("template_inlier_chord", .06)),
            spin_update_interval_us=int(self.config.get(
                "spin_update_interval_us", 5000)))
        self.buffers = [np.empty(0, dtype=EVENT_DTYPE), np.empty(0, dtype=EVENT_DTYPE)]
        if self.config.get("warmup_event_kernel", True):
            dummy = np.zeros(1, dtype=EVENT_DTYPE)
            motion_compensated_ball_patch(
                dummy, (0.0, 0.0), 1.0, int(self.config["patch_size"]),
                0, 1.0, (0.0, 0.0), 1)

    def _project(self, points_m: np.ndarray, camera: int) -> np.ndarray:
        if camera == 0:
            rvec = np.zeros(3); tvec = np.zeros(3)
            matrix, distortion = self.calibration.K1, self.calibration.D1
        else:
            rvec, _ = cv2.Rodrigues(self.calibration.R)
            tvec = self.calibration.T.reshape(3)
            matrix, distortion = self.calibration.K2, self.calibration.D2
        projected, _ = cv2.projectPoints(
            np.asarray(points_m, np.float64), rvec, tvec, matrix, distortion)
        return projected.reshape(-1, 2)

    def _guided_directions(self, timestamp: int, image_points, center_m: np.ndarray):
        tracker = self.stream.tracker
        if tracker.pose is None or tracker.timestamp_us is None:
            return None
        gap_s = (timestamp - tracker.timestamp_us) * 1e-6
        if gap_s <= 0 or gap_s > self.stream.reset_gap_us * 1e-6:
            return None
        prediction = tracker.pose
        if tracker.omega is not None:
            delta, _ = cv2.Rodrigues(tracker.omega * gap_s)
            prediction = delta @ prediction
        directions = (prediction @ tracker.template.T).T
        points_m = (np.asarray(center_m, np.float64)[None, :]
                    + float(self.config["ball_radius_m"]) * directions)
        projected = [self._project(points_m, camera) for camera in (0, 1)]
        gate = float(self.config.get("guided_marker_gate_px", 12.0))
        assignments = []
        for expected, observed in zip(projected, image_points):
            if not len(observed):
                return (np.empty(0, np.int32), np.empty((0, 3), np.float64))
            distance = np.linalg.norm(expected[:, None, :] - observed[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(distance)
            valid = distance[rows, cols] <= gate
            assignments.append({int(row): int(col) for row, col in zip(rows[valid], cols[valid])})
        common = sorted(set(assignments[0]) & set(assignments[1]))
        ray_tables = [self._ray_sphere_direction_table(
            image_points[camera], camera, center_m) for camera in (0, 1)]
        accepted_ids = []
        accepted = []
        for identifier in common:
            direction = self._fuse_direction_estimates(
                ray_tables[0][assignments[0][identifier]],
                ray_tables[1][assignments[1][identifier]])
            if direction is None or np.linalg.norm(
                    direction - directions[identifier]) > float(
                        self.config.get("guided_direction_chord", .20)):
                continue
            accepted_ids.append(identifier)
            accepted.append(direction)
        return (np.asarray(accepted_ids, np.int32),
                np.asarray(accepted, np.float64).reshape(-1, 3))

    def _score_projected_pose(self, rotation: np.ndarray, image_points,
                              center_m: np.ndarray):
        """Score one pose by shared template identities in both cameras."""
        directions = (rotation @ self.stream.tracker.template.T).T
        points_m = (center_m[None, :]
                    + float(self.config["ball_radius_m"]) * directions)
        camera_points = (
            points_m,
            (self.calibration.R @ points_m.T).T + self.calibration.T.reshape(1, 3),
        )
        camera_normals = (directions, (self.calibration.R @ directions.T).T)
        visible = np.ones(len(directions), bool)
        for points_camera, normals_camera in zip(camera_points, camera_normals):
            visible &= np.sum(normals_camera * (-points_camera), axis=1) > 0
        visible_ids = np.flatnonzero(visible)
        if len(visible_ids) < self.stream.minimum_markers:
            return None
        projected = [self._project(points_m, camera) for camera in (0, 1)]
        gate = float(self.config.get("acquisition_marker_gate_px", 4.0))
        assignments = []
        residuals = []
        for expected, observed in zip(projected, image_points):
            distance = np.linalg.norm(
                expected[visible_ids, None, :] - observed[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(distance)
            valid = distance[rows, cols] <= gate
            assignments.append({int(visible_ids[row]): int(col)
                                for row, col in zip(rows[valid], cols[valid])})
            residuals.append({int(visible_ids[row]): float(distance[row, col])
                              for row, col in zip(rows[valid], cols[valid])})
        common = sorted(set(assignments[0]) & set(assignments[1]))
        if len(common) < self.stream.minimum_markers:
            return None
        pixel_residual = float(np.mean([
            residuals[0][identifier] + residuals[1][identifier]
            for identifier in common]))
        return common, assignments, pixel_residual

    def _acquire_projected_pose(self, image_points, center_m: np.ndarray):
        """Resolve anonymous stereo points by scoring full poses in both images."""
        best_key = (-1, -float("inf")); best = None
        limit = int(self.config.get("acquisition_pose_hypotheses", 30))
        for camera in (0, 1):
            ray_directions = self._ray_sphere_directions(
                image_points[camera], camera, center_m)
            for rotation in self.stream.tracker.matcher.rotation_hypotheses(
                    ray_directions, maximum_hypotheses=limit // 2):
                scored = self._score_projected_pose(rotation, image_points, center_m)
                if scored is None:
                    continue
                common, assignments, residual = scored
                key = (len(common), -residual)
                if key > best_key:
                    best_key = key; best = (common, assignments)
        if best is None:
            empty = np.empty(0, np.int32)
            return empty, np.empty((0, 3), np.float64)
        common, assignments = best
        ray_tables = [self._ray_sphere_direction_table(
            image_points[camera], camera, center_m) for camera in (0, 1)]
        accepted_ids = []; accepted_directions = []
        for identifier in common:
            direction = self._fuse_direction_estimates(
                ray_tables[0][assignments[0][identifier]],
                ray_tables[1][assignments[1][identifier]])
            if direction is not None:
                accepted_ids.append(identifier); accepted_directions.append(direction)
        return (np.asarray(accepted_ids, np.int32),
                np.asarray(accepted_directions, np.float64).reshape(-1, 3))

    def _ray_sphere_directions(self, image_points: np.ndarray, camera: int,
                               center_m: np.ndarray) -> np.ndarray:
        """Lift visible image dots to the near intersection with the known ball."""
        table = self._ray_sphere_direction_table(image_points, camera, center_m)
        return table[np.isfinite(table).all(axis=1)]

    def _ray_sphere_direction_table(self, image_points: np.ndarray, camera: int,
                                    center_m: np.ndarray) -> np.ndarray:
        """Return one lifted direction per pixel, using NaNs for missed rays."""
        pixels = np.asarray(image_points, np.float64).reshape(-1, 1, 2)
        if not len(pixels):
            return np.empty((0, 3), np.float64)
        if camera == 0:
            matrix, distortion = self.calibration.K1, self.calibration.D1
            origin = np.zeros(3)
            camera_to_master = np.eye(3)
        else:
            matrix, distortion = self.calibration.K2, self.calibration.D2
            camera_to_master = self.calibration.R.T
            origin = -camera_to_master @ self.calibration.T.reshape(3)
        normalized = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 2)
        rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
        rays = (camera_to_master @ rays_camera.T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        offset = origin - np.asarray(center_m, np.float64)
        projection = rays @ offset
        discriminant = projection ** 2 - (
            float(offset @ offset) - float(self.config["ball_radius_m"]) ** 2)
        valid = discriminant >= 0
        distance = -projection[valid] - np.sqrt(np.maximum(discriminant[valid], 0.0))
        valid_distance = distance > 0
        points = origin + distance[valid_distance, None] * rays[valid][valid_distance]
        directions = points - center_m
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        table = np.full((len(pixels), 3), np.nan, np.float64)
        valid_indices = np.flatnonzero(valid)[valid_distance]
        table[valid_indices] = directions
        return table

    @staticmethod
    def _fuse_direction_estimates(left: np.ndarray,
                                  right: np.ndarray) -> Optional[np.ndarray]:
        estimates = [value for value in (left, right) if np.isfinite(value).all()]
        if not estimates:
            return None
        fused = np.sum(estimates, axis=0)
        norm = float(np.linalg.norm(fused))
        return None if norm < 1e-12 else fused / norm

    def reset(self) -> None:
        self.stream.reset()
        self.buffers = [np.empty(0, dtype=EVENT_DTYPE), np.empty(0, dtype=EVENT_DTYPE)]

    def _append(self, camera: int, events: np.ndarray, timestamp_us: int) -> np.ndarray:
        if events is not None and len(events):
            self.buffers[camera] = np.concatenate((self.buffers[camera], events))
        minimum = int(timestamp_us) - int(self.config["window_us"])
        if len(self.buffers[camera]):
            self.buffers[camera] = self.buffers[camera][
                self.buffers[camera]["timestamp"].astype(np.int64) >= minimum]
        return self.buffers[camera]

    def process(self, timestamp_us: int, master_events: np.ndarray, slave_events: np.ndarray,
                master_track: Optional[Track2D], slave_track: Optional[Track2D],
                center_m: Optional[np.ndarray]) -> StereoTemplateSpinChannelResult:
        timestamp = int(timestamp_us)
        buffers = (self._append(0, master_events, timestamp),
                   self._append(1, slave_events, timestamp))
        if master_track is None or slave_track is None or center_m is None:
            return StereoTemplateSpinChannelResult(timestamp, 0, 0, 0, None)
        tracks = (master_track, slave_track)
        points = []
        counts = []
        size = int(self.config["patch_size"])
        for events, track, detector in zip(buffers, tracks, self.detectors):
            patch = motion_compensated_ball_patch(
                events, track.point, track.radius, size, timestamp,
                float(self.config["tau_us"]), (track.vx, track.vy),
                int(self.config["window_us"]))
            gray = detector._gray(patch)
            dots = detector._dot_centers(gray)
            image_points = detector._patch_to_image(dots[:, :2], track) if len(dots) else np.empty((0, 2))
            points.append(image_points); counts.append(len(image_points))
        center = np.asarray(center_m, np.float64)
        guided = self._guided_directions(timestamp, points, center)
        if guided is None or len(guided[0]) < self.stream.minimum_markers:
            identifiers, directions = self._acquire_projected_pose(points, center)
            stream = self.stream.update_corresponded(
                timestamp, identifiers, directions)
        else:
            identifiers, directions = guided
            stream = self.stream.update_corresponded(timestamp, identifiers, directions)
        return StereoTemplateSpinChannelResult(
            timestamp, counts[0], counts[1], len(directions), stream)
