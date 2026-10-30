from __future__ import annotations

from pathlib import Path

import numpy as np


class TableFrame:
    """Rigid transform from the master-camera frame to table coordinates."""

    def __init__(self, path: str | Path):
        data = np.load(Path(path))
        self.R = np.asarray(data["R_table_from_master"], np.float64).reshape(3, 3)
        self.origin = np.asarray(data["origin_master_m"], np.float64).reshape(3)
        if not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-5):
            raise ValueError("球桌坐标旋转矩阵不是正交矩阵")
        if np.linalg.det(self.R) < 0.999:
            raise ValueError("球桌坐标必须是右手坐标系")

    def point(self, point_master_m) -> np.ndarray:
        return self.R @ (np.asarray(point_master_m, np.float64) - self.origin)

    def vector(self, vector_master) -> np.ndarray:
        return self.R @ np.asarray(vector_master, np.float64)


def spin_components_table(omega_table_rad_s, velocity_table_mps) -> dict:
    """Resolve spin around flight-relative topspin, vertical and gyro axes."""
    omega = np.asarray(omega_table_rad_s, np.float64).reshape(3)
    velocity = np.asarray(velocity_table_mps, np.float64).reshape(3)
    horizontal = velocity.copy(); horizontal[2] = 0.0
    speed = float(np.linalg.norm(horizontal))
    if speed < 1e-6:
        raise ValueError("水平球速过低，无法定义上旋方向")
    flight = horizontal / speed
    up = np.array([0.0, 0.0, 1.0])
    topspin_axis = np.cross(up, flight)
    axes = np.vstack((topspin_axis, up, flight))
    components_rad_s = axes @ omega
    components_rpm = components_rad_s * 60.0 / (2.0 * np.pi)
    names = (
        "topspin" if components_rpm[0] >= 0 else "backspin",
        "left_sidespin" if components_rpm[1] >= 0 else "right_sidespin",
        "positive_gyrospin" if components_rpm[2] >= 0 else "negative_gyrospin",
    )
    dominant = int(np.argmax(np.abs(components_rpm)))
    return {
        "flight_direction_table_xy": flight.tolist(),
        "topspin_axis_table": topspin_axis.tolist(),
        "topspin_signed_rpm": float(components_rpm[0]),
        "vertical_sidespin_signed_rpm": float(components_rpm[1]),
        "gyrospin_signed_rpm": float(components_rpm[2]),
        "dominant_spin": names[dominant],
        "dominant_component_rpm": float(abs(components_rpm[dominant])),
        "sign_convention": {
            "topspin_positive": "forward no-slip rolling direction",
            "vertical_sidespin_positive": "right-hand rotation about table +Z",
            "gyrospin_positive": "right-hand rotation about horizontal flight direction",
        },
    }
