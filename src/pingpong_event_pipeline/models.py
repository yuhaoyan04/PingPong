from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection2D:
    timestamp_us: int
    x: float
    y: float
    radius: float
    score: float
    event_count: int

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class Track2D:
    timestamp_us: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    confidence: float
    measured: bool

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class StereoState:
    timestamp_us: int
    position_m: np.ndarray
    velocity_mps: np.ndarray
    reprojection_error_px: float
    left: Track2D
    right: Track2D


@dataclass
class SpinEstimate:
    timestamp_us: int
    omega_camera_rad_s: np.ndarray
    rpm: float
    feature_count: int
    quality: float
    omega_world_rad_s: Optional[np.ndarray] = field(default=None)
