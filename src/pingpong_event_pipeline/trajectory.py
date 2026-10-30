from __future__ import annotations

from typing import Optional

import numpy as np


class TrajectoryFilter3D:
    """3D position/velocity EKF with gravity and quadratic drag."""

    def __init__(self, config: dict):
        if "gravity_vector_camera_mps2" in config:
            self.gravity_vector = np.asarray(config["gravity_vector_camera_mps2"], dtype=np.float64)
        else:
            self.gravity_vector = np.array([0.0, 0.0, -float(config["gravity_mps2"])])
        self.drag = float(config["drag_coefficient"])
        self.process_noise = float(config["process_noise"])
        self.measurement_noise = float(config["measurement_noise_m"])
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6) * 1e3
        self.last_us: Optional[int] = None
        self.initialized = False

    def _propagate(self, dt: float) -> None:
        p, v = self.state[:3], self.state[3:]
        speed = np.linalg.norm(v)
        acceleration = self.gravity_vector - self.drag * speed * v
        self.state[:3] = p + v * dt + 0.5 * acceleration * dt * dt
        self.state[3:] = v + acceleration * dt
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        if speed > 1e-6:
            drag_jacobian = -self.drag * (speed * np.eye(3) + np.outer(v, v) / speed)
            F[3:, 3:] += drag_jacobian * dt
        Q = np.eye(6) * self.process_noise * max(dt, 1e-3)
        self.P = F @ self.P @ F.T + Q

    def update(self, timestamp_us: int, position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position_m = np.asarray(position_m, dtype=np.float64)
        if not self.initialized:
            self.state[:3] = position_m
            self.P = np.diag([1e-4] * 3 + [25.0] * 3)
            self.last_us = timestamp_us
            self.initialized = True
            return self.state[:3].copy(), self.state[3:].copy()
        dt = max((timestamp_us - self.last_us) * 1e-6, 1e-5)
        self._propagate(dt)
        self.last_us = timestamp_us
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        R = np.eye(3) * self.measurement_noise ** 2
        innovation = position_m - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.state[:3].copy(), self.state[3:].copy()

    def predict_position(self, horizon_s: float, steps: int = 20) -> np.ndarray:
        state, covariance = self.state.copy(), self.P.copy()
        last = self.last_us
        dt = horizon_s / max(steps, 1)
        for _ in range(steps):
            self._propagate(dt)
        prediction = self.state[:3].copy()
        self.state, self.P, self.last_us = state, covariance, last
        return prediction
