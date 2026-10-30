"""Track a fast event-camera ball from complementary ON/OFF crescents."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Detection2D


@dataclass
class _Lobe:
    polarity: int
    center: np.ndarray
    points: np.ndarray
    support: int
    peak: float


def _robust_lobe_circle(off: _Lobe, on: _Lobe) -> tuple[float, float, float]:
    """Estimate the silhouette without letting sparse trail points set its size.

    The old minimum-enclosing circle was determined by the single most distant
    event.  At table-tennis speed that event is often part of a motion trail or
    background structure.  Quantiles in the lobe-separation coordinate system
    retain the dense ball silhouette and ignore those extreme points.
    """
    direction = on.center - off.center
    length = float(np.linalg.norm(direction))
    axis = direction / max(length, 1e-6)
    perpendicular = np.array([-axis[1], axis[0]], np.float64)
    points = np.vstack((off.points, on.points)).astype(np.float64)
    along = points @ axis
    across = points @ perpendicular
    along_bounds = np.percentile(along, (5.0, 95.0))
    across_bounds = np.percentile(across, (5.0, 95.0))
    center = (axis * float(np.mean(along_bounds))
              + perpendicular * float(np.mean(across_bounds)))
    radius = max(float(np.ptp(along_bounds)), float(np.ptp(across_bounds))) * .5
    return float(center[0]), float(center[1]), radius


def _polarity_lobes(events: np.ndarray, roi: tuple[int, int, int, int],
                    polarity: int, sigma_px: float, threshold_fraction: float,
                    minimum_support: int) -> list[_Lobe]:
    x0, y0, x1, y1 = roi
    selected = events[
        (events["polarity"] == polarity)
        & (events["x"] >= x0) & (events["x"] < x1)
        & (events["y"] >= y0) & (events["y"] < y1)
    ]
    if len(selected) < minimum_support:
        return []
    height, width = y1 - y0, x1 - x0
    counts = np.zeros((height, width), np.float32)
    xx = selected["x"].astype(np.intp) - x0
    yy = selected["y"].astype(np.intp) - y0
    np.add.at(counts, (yy, xx), 1.0)
    smooth = cv2.GaussianBlur(counts, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
    peak = float(smooth.max())
    if peak <= 0:
        return []
    mask = (smooth >= max(0.02, threshold_fraction * peak)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    lobes: list[_Lobe] = []
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] < 12:
            continue
        inside = labels[yy, xx] == label
        support = int(np.count_nonzero(inside))
        if support < minimum_support:
            continue
        points = np.column_stack((xx[inside] + x0, yy[inside] + y0)).astype(np.float64)
        weights = counts[labels == label]
        lobes.append(_Lobe(
            polarity, centroids[label] + [x0, y0], points, support,
            float(weights.max()) if len(weights) else peak,
        ))
    lobes.sort(key=lambda value: value.support, reverse=True)
    return lobes[:8]


class CrescentBallDetector:
    """Estimate a ball circle from spatially separated polarity lobes.

    A translating ball often produces two incomplete crescents rather than a
    complete event circle. Pairing opposite-polarity lobes and enclosing their
    supported event pixels is more stable than requiring a Hough circle.
    """

    def __init__(self, search_roi: tuple[int, int, int, int],
                 min_radius_px: float = 15.0, max_radius_px: float = 140.0,
                 sigma_px: float = 4.0, threshold_fraction: float = 0.16,
                 minimum_lobe_events: int = 30, maximum_lobe_separation_px: float = 240.0):
        self.search_roi = tuple(map(int, search_roi))
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)
        self.sigma_px = float(sigma_px)
        self.threshold_fraction = float(threshold_fraction)
        self.minimum_lobe_events = int(minimum_lobe_events)
        self.maximum_lobe_separation_px = float(maximum_lobe_separation_px)

    def detect(self, events: np.ndarray, timestamp_us: int,
               predicted: np.ndarray | None = None) -> list[Detection2D]:
        lobes = [
            _polarity_lobes(events, self.search_roi, polarity, self.sigma_px,
                            self.threshold_fraction, self.minimum_lobe_events)
            for polarity in (0, 1)
        ]
        candidates: list[Detection2D] = []
        for off in lobes[0]:
            for on in lobes[1]:
                separation = float(np.linalg.norm(off.center - on.center))
                if not 3.0 <= separation <= self.maximum_lobe_separation_px:
                    continue
                cx, cy, radius = _robust_lobe_circle(off, on)
                if not self.min_radius_px <= radius <= self.max_radius_px:
                    continue
                balance = min(off.support, on.support) / max(off.support, on.support)
                support_score = min(1.0, (off.support + on.support) / 1200.0)
                score = 0.55 * balance + 0.35 * support_score + 0.10 * min(1.0, separation / radius)
                if predicted is not None:
                    prediction = np.asarray(predicted, np.float64).reshape(-1)
                    center_error = np.linalg.norm(np.array([cx, cy]) - prediction[:2])
                    radius_error = abs(radius - prediction[2]) if len(prediction) >= 3 else 0.0
                    score -= 0.003 * center_error + 0.004 * radius_error
                candidates.append(Detection2D(
                    int(timestamp_us), float(cx), float(cy), float(radius),
                    float(score), int(off.support + on.support)))
        candidates.sort(key=lambda value: value.score, reverse=True)
        return candidates
