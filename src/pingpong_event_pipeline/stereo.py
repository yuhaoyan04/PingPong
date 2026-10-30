from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import Detection2D, Track2D


class StereoCalibration:
    def __init__(self, path: str | Path):
        data = np.load(Path(path))
        required = ("K1", "D1", "K2", "D2", "R", "T", "P1", "P2", "F")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"标定文件缺少: {missing}")
        for key in required:
            setattr(self, key, data[key].astype(np.float64))
        self.R1 = data["R1"].astype(np.float64) if "R1" in data else np.eye(3)
        self.R2 = data["R2"].astype(np.float64) if "R2" in data else np.eye(3)

    def epipolar_error(self, left_xy: np.ndarray, right_xy: np.ndarray) -> float:
        x1 = np.array([left_xy[0], left_xy[1], 1.0])
        x2 = np.array([right_xy[0], right_xy[1], 1.0])
        line = self.F @ x1
        return float(abs(x2 @ line) / max(np.hypot(line[0], line[1]), 1e-12))

    def symmetric_epipolar_error(self, left_xy: np.ndarray, right_xy: np.ndarray) -> float:
        x1 = np.array([left_xy[0], left_xy[1], 1.0])
        x2 = np.array([right_xy[0], right_xy[1], 1.0])
        value = abs(float(x2 @ self.F @ x1))
        line2 = self.F @ x1
        line1 = self.F.T @ x2
        d1 = value / max(float(np.hypot(line1[0], line1[1])), 1e-12)
        d2 = value / max(float(np.hypot(line2[0], line2[1])), 1e-12)
        return 0.5 * (d1 + d2)

    def triangulate(self, left_xy: np.ndarray, right_xy: np.ndarray) -> Tuple[np.ndarray, float]:
        left_xy = np.asarray(left_xy, dtype=np.float64)
        right_xy = np.asarray(right_xy, dtype=np.float64)
        left = cv2.undistortPoints(left_xy.reshape(1, 1, 2), self.K1, self.D1,
                                   R=self.R1, P=self.P1[:, :3])
        right = cv2.undistortPoints(right_xy.reshape(1, 1, 2), self.K2, self.D2,
                                    R=self.R2, P=self.P2[:, :3])
        homogeneous = cv2.triangulatePoints(self.P1, self.P2, left.reshape(2, 1), right.reshape(2, 1))
        point_rectified = (homogeneous[:3, 0] / homogeneous[3, 0]).astype(np.float64)
        # P1/P2 describe the rectified stereo pair. Convert back to the original
        # master-camera coordinate system used by R/T and by the trajectory code.
        point = self.R1.T @ point_rectified
        error = self.reprojection_error(point, left_xy, right_xy)
        return point, error

    def reprojection_error(self, point: np.ndarray, left_xy: np.ndarray, right_xy: np.ndarray) -> float:
        # triangulate() 返回第一台相机坐标系中的点；用原始相机模型回投影。
        lp, _ = cv2.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3), self.K1, self.D1)
        rvec, _ = cv2.Rodrigues(self.R)
        rp, _ = cv2.projectPoints(point.reshape(1, 3), rvec, self.T.reshape(3), self.K2, self.D2)
        lp, rp = lp.reshape(2), rp.reshape(2)
        return float(0.5 * (np.linalg.norm(lp - left_xy) + np.linalg.norm(rp - right_xy)))


def match_tracks(
    left: Optional[Track2D],
    right: Optional[Track2D],
    calibration: StereoCalibration,
    max_epipolar_error_px: float,
) -> Optional[Tuple[Track2D, Track2D]]:
    if left is None or right is None:
        return None
    if abs(left.timestamp_us - right.timestamp_us) > 2_000:
        return None
    if calibration.epipolar_error(left.point, right.point) > max_epipolar_error_px:
        return None
    return left, right


def select_stereo_detection_pair(
    left: Sequence[Detection2D],
    right: Sequence[Detection2D],
    calibration: StereoCalibration,
    max_epipolar_error_px: float,
    max_radius_ratio: float = 2.0,
    ball_radius_m: Optional[float] = None,
    minimum_depth_m: float = 0.3,
    maximum_depth_m: float = 3.0,
    max_radius_scale_error: float = 0.8,
    left_prediction: Optional[np.ndarray] = None,
    right_prediction: Optional[np.ndarray] = None,
    maximum_prediction_error_px: float = 60.0,
    previous_point_m: Optional[np.ndarray] = None,
    maximum_position_jump_m: Optional[float] = None,
) -> Optional[Tuple[Detection2D, Detection2D]]:
    """Choose the jointly plausible ball pair before either tracker is updated."""
    best = None
    best_cost = float("inf")
    for ld in left:
        for rd in right:
            prediction_cost = 0.0
            if left_prediction is not None:
                error = float(np.linalg.norm(ld.point - np.asarray(left_prediction)[:2]))
                if error > maximum_prediction_error_px:
                    continue
                prediction_cost += error / max(maximum_prediction_error_px, 1e-6)
            if right_prediction is not None:
                error = float(np.linalg.norm(rd.point - np.asarray(right_prediction)[:2]))
                if error > maximum_prediction_error_px:
                    continue
                prediction_cost += error / max(maximum_prediction_error_px, 1e-6)
            radius_ratio = max(ld.radius, rd.radius) / max(min(ld.radius, rd.radius), 1e-6)
            if radius_ratio > max_radius_ratio:
                continue
            epi = calibration.symmetric_epipolar_error(ld.point, rd.point)
            if epi > max_epipolar_error_px:
                continue
            physical_cost = 0.0
            point = None
            if ball_radius_m is not None:
                point, _ = calibration.triangulate(ld.point, rd.point)
                right_point = calibration.R @ point + calibration.T.reshape(3)
                if (not np.isfinite(point).all() or point[2] < minimum_depth_m or
                        point[2] > maximum_depth_m or right_point[2] < minimum_depth_m or
                        right_point[2] > maximum_depth_m):
                    continue
                expected_left = calibration.K1[0, 0] * ball_radius_m / point[2]
                expected_right = calibration.K2[0, 0] * ball_radius_m / right_point[2]
                left_error = abs(float(np.log(ld.radius / max(expected_left, 1e-6))))
                right_error = abs(float(np.log(rd.radius / max(expected_right, 1e-6))))
                if left_error > max_radius_scale_error or right_error > max_radius_scale_error:
                    continue
                physical_cost = left_error + right_error
            continuity_cost = 0.0
            if previous_point_m is not None and maximum_position_jump_m is not None:
                if point is None:
                    point, _ = calibration.triangulate(ld.point, rd.point)
                jump = float(np.linalg.norm(point - np.asarray(previous_point_m)))
                if jump > maximum_position_jump_m:
                    continue
                continuity_cost = jump / max(maximum_position_jump_m, 1e-6)
            cost = (epi / max(max_epipolar_error_px, 1e-6) + physical_cost + prediction_cost +
                    continuity_cost +
                    abs(np.log(radius_ratio)) + 0.5 * (2.0 - ld.score - rd.score))
            if cost < best_cost:
                best_cost = cost
                best = (ld, rd)
    return best
