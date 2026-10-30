from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .models import Detection2D, Track2D


class BallTracker2D:
    """Constant-velocity Kalman filter with gated candidate association."""

    def __init__(self, max_missed: int = 12):
        self.x = np.zeros(5, dtype=np.float64)  # x, y, vx, vy, radius
        self.P = np.eye(5) * 1e3
        self.last_us: Optional[int] = None
        self.missed = 0
        self.max_missed = max_missed
        self.initialized = False

    def predict(self, timestamp_us: int) -> Optional[np.ndarray]:
        if not self.initialized or self.last_us is None:
            return None
        dt = max((timestamp_us - self.last_us) * 1e-6, 0.0)
        F = np.eye(5)
        F[0, 2] = dt
        F[1, 3] = dt
        q = max(dt, 1e-3)
        # Pixel velocity is expressed per second. A table-tennis ball commonly
        # crosses several pixels per millisecond, so the velocity covariance
        # must be large enough for the second observation to establish speed.
        Q = np.diag([4, 4, 2.0e6, 2.0e6, 1]) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_us = timestamp_us
        return self.x.copy()

    def update(self, timestamp_us: int, detections: Iterable[Detection2D]) -> Optional[Track2D]:
        prediction = self.predict(timestamp_us)
        candidates = list(detections)
        measured = False
        selected = None
        if not self.initialized and candidates:
            selected = max(candidates, key=lambda d: d.score)
            self.x[:] = [selected.x, selected.y, 0, 0, selected.radius]
            self.P = np.diag([16, 16, 1e7, 1e7, 9]).astype(float)
            self.last_us = timestamp_us
            self.initialized = True
            measured = True
        elif self.initialized and candidates:
            gate = max(35.0, 4.0 * np.sqrt(max(self.P[0, 0], self.P[1, 1])))
            selected = min(candidates, key=lambda d: np.linalg.norm(d.point - self.x[:2]) - 20 * d.score)
            if np.linalg.norm(selected.point - self.x[:2]) <= gate:
                z = np.array([selected.x, selected.y, selected.radius])
                H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 4] = 1
                R = np.diag([9, 9, 4]).astype(float)
                innovation = z - H @ self.x
                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x += K @ innovation
                self.P = (np.eye(5) - K @ H) @ self.P
                measured = True
        self.missed = 0 if measured else self.missed + 1
        if self.missed > self.max_missed:
            self.initialized = False
            return None
        if not self.initialized:
            return None
        confidence = selected.score if measured and selected is not None else max(0.0, 0.5 - 0.04 * self.missed)
        return Track2D(timestamp_us, *self.x, confidence, measured)

    def advance(self, timestamp_us: int) -> Optional[Track2D]:
        """Advance through a frame where detection was intentionally not run."""
        self.predict(timestamp_us)
        if not self.initialized:
            return None
        confidence = max(0.05, 0.5 - 0.04 * self.missed)
        return Track2D(timestamp_us, *self.x, confidence, False)
