from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .models import Detection2D


class ClassicalBallDetector:
    def __init__(self, config: dict):
        self.cfg = config

    def detect(
        self,
        surface: np.ndarray,
        timestamp_us: int,
        predicted_xy: Optional[Tuple[float, float]] = None,
    ) -> List[Detection2D]:
        activity = np.maximum(surface[0], surface[1])
        image = np.clip(activity * 255, 0, 255).astype(np.uint8)
        x0 = y0 = 0
        if predicted_xy is not None:
            half = int(self.cfg["roi_half_size_px"])
            cx, cy = map(int, predicted_xy)
            x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
            y0, y1 = max(0, cy - half), min(image.shape[0], cy + half)
            crop = image[y0:y1, x0:x1]
        else:
            crop = image
            roi = self.cfg.get("search_roi")
            if roi is not None:
                x0, y0, x1, y1 = map(int, roi)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
                crop = image[y0:y1, x0:x1]
        if crop.size == 0 or not np.any(crop):
            return []
        nonzero = crop[crop > 0]
        threshold = max(
            int(self.cfg["minimum_threshold"]),
            int(np.percentile(nonzero, float(self.cfg["threshold_percentile"]))) if nonzero.size else 255,
        )
        # percentile 常会等于最亮事件值；使用 >=，避免 OpenCV 的严格 > 把全部事件清空。
        mask = (crop >= threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection2D] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.cfg["min_area_px"] <= area <= self.cfg["max_area_px"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > self.cfg["max_aspect_ratio"]:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not self.cfg["min_radius_px"] <= radius <= self.cfg["max_radius_px"]:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
            fill = area / max(np.pi * radius * radius, 1.0)
            local = crop[y:y + h, x:x + w]
            event_count = int(np.count_nonzero(local))
            score = float(np.clip(0.45 * circularity + 0.35 * fill + 0.20 * min(event_count / 80, 1), 0, 1))
            detections.append(Detection2D(timestamp_us, cx + x0, cy + y0, radius, score, event_count))
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections


class HoughCircleBallDetector:
    """Generate ball candidates from incomplete circular event arcs."""

    def __init__(self, config: dict):
        self.cfg = config

    @staticmethod
    def _refine_circle(image: np.ndarray, cx: float, cy: float, radius: float,
                       iterations: int = 5) -> tuple[float, float, float]:
        """Refine a Hough proposal on full-resolution event-ring pixels."""
        margin = max(3.0, 0.14 * float(radius))
        x0 = max(0, int(np.floor(cx - radius - margin)))
        x1 = min(image.shape[1], int(np.ceil(cx + radius + margin + 1)))
        y0 = max(0, int(np.floor(cy - radius - margin)))
        y1 = min(image.shape[0], int(np.ceil(cy + radius + margin + 1)))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return float(cx), float(cy), float(radius)
        yy, xx = np.nonzero(crop >= max(12, np.percentile(crop[crop > 0], 35)
                                        if np.any(crop > 0) else 255))
        x = xx.astype(np.float64) + x0; y = yy.astype(np.float64) + y0
        radial = np.hypot(x - cx, y - cy)
        keep = np.abs(radial - radius) <= margin
        x, y = x[keep], y[keep]
        if len(x) < 16:
            return float(cx), float(cy), float(radius)
        center = np.array([cx, cy], np.float64); refined_radius = float(radius)
        weights = np.ones(len(x), np.float64)
        for _ in range(max(1, int(iterations))):
            A = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
            b = x * x + y * y
            root_w = np.sqrt(weights)
            solution = np.linalg.lstsq(A * root_w[:, None], b * root_w, rcond=None)[0]
            candidate_center = solution[:2]
            candidate_radius = np.sqrt(max(
                solution[2] + np.dot(candidate_center, candidate_center), 1.0))
            residual = np.abs(np.hypot(
                x - candidate_center[0], y - candidate_center[1]) - candidate_radius)
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 0.25
            weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
            center, refined_radius = candidate_center, float(candidate_radius)
        # Do not allow a partial unrelated arc to drag refinement far from Hough.
        if (np.linalg.norm(center - np.array([cx, cy])) > margin or
                abs(refined_radius - radius) > margin):
            return float(cx), float(cy), float(radius)
        return float(center[0]), float(center[1]), refined_radius

    @staticmethod
    def _ring_score(image: np.ndarray, cx: float, cy: float, radius: float) -> float:
        return float(HoughCircleBallDetector._ring_scores(
            image, np.array([[cx, cy, radius]], np.float64))[0])

    @staticmethod
    def _ring_scores(image: np.ndarray, circles: np.ndarray, samples: int = 96) -> np.ndarray:
        """Score many circle hypotheses in one NumPy operation."""
        circles = np.asarray(circles, np.float64).reshape(-1, 3)
        if not len(circles):
            return np.empty(0, np.float64)
        angles = np.linspace(0, 2 * np.pi, max(24, int(samples)), endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        radii = circles[:, 2, None, None] + np.array([-2.0, 0.0, 2.0])[None, :, None]
        x = np.rint(circles[:, 0, None, None] + radii * cos_a[None, None, :]).astype(np.intp)
        y = np.rint(circles[:, 1, None, None] + radii * sin_a[None, None, :]).astype(np.intp)
        valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        values = np.zeros(x.shape, np.uint8)
        values[valid] = image[y[valid], x[valid]]
        ring = values.max(axis=1)
        coverage = np.count_nonzero(ring >= 18, axis=1) / ring.shape[1]
        strength = np.mean(ring, axis=1) / 255.0
        return np.clip(0.75 * coverage + 0.25 * strength, 0.0, 1.0)

    def _search_ring_candidates_near(self, image: np.ndarray, cx: float, cy: float,
                                     radius: float, maximum: int) -> list[tuple[float, float, float, float]]:
        """Return distinct local ring maxima around a predicted ball."""
        half = int(self.cfg.get("predictive_search_px", 16))
        step = max(1, int(self.cfg.get("predictive_search_step_px", 4)))
        radius_half = int(self.cfg.get("predictive_radius_search_px", 4))
        radius_step = max(1, int(self.cfg.get("predictive_radius_step_px", 4)))
        dx = np.arange(-half, half + 1, step, dtype=np.float64)
        dy = np.arange(-half, half + 1, step, dtype=np.float64)
        dr = np.arange(-radius_half, radius_half + 1, radius_step, dtype=np.float64)
        gy, gx, gr = np.meshgrid(dy, dx, dr, indexing="ij")
        circles = np.column_stack((cx + gx.ravel(), cy + gy.ravel(),
                                   np.maximum(2.0, radius + gr.ravel())))
        scores = self._ring_scores(
            image, circles, int(self.cfg.get("predictive_ring_samples", 96)))
        order = np.argsort(scores)[::-1]
        selected: list[tuple[float, float, float, float]] = []
        nms_distance = float(self.cfg.get("predictive_nms_px", max(3, step)))
        for index in order:
            candidate = circles[index]
            if any(np.linalg.norm(candidate[:2] - np.asarray(item[:2])) < nms_distance and
                   abs(candidate[2] - item[2]) < radius_step for item in selected):
                continue
            if self.cfg.get("predictive_refine_candidates", True):
                refined = self._refine_circle(
                    image, *candidate,
                    int(self.cfg.get("hough_refine_iterations", 5)))
                score = max(float(scores[index]), self._ring_score(image, *refined))
            else:
                refined = tuple(candidate)
                score = float(scores[index])
            selected.append((*refined, score))
            if len(selected) >= maximum:
                break
        return selected

    def _search_ring_near(self, image: np.ndarray, cx: float, cy: float,
                          radius: float) -> tuple[float, float, float, float]:
        """Search a bounded neighborhood so prediction updates can follow motion."""
        return self._search_ring_candidates_near(image, cx, cy, radius, 1)[0]

    def detect(
        self,
        surface: np.ndarray,
        timestamp_us: int,
        predicted_xy: Optional[Tuple[float, float]] = None,
    ) -> List[Detection2D]:
        image = np.clip(np.maximum(surface[0], surface[1]) * 255, 0, 255).astype(np.uint8)
        x0 = y0 = 0
        prior_radius = None
        crop = image
        if predicted_xy is not None:
            half = int(self.cfg.get("roi_half_size_px", 180))
            prediction = np.asarray(predicted_xy, dtype=float).reshape(-1)
            cx, cy = map(int, prediction[:2])
            prior_radius = float(prediction[2]) if len(prediction) >= 3 else None
            x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
            y0, y1 = max(0, cy - half), min(image.shape[0], cy + half)
            crop = image[y0:y1, x0:x1]
        else:
            roi = self.cfg.get("search_roi")
            if roi is not None:
                x0, y0, x1, y1 = map(int, roi)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
                crop = image[y0:y1, x0:x1]
        if crop.size == 0 or np.count_nonzero(crop) < 20:
            return []
        # Once a track exists, update its known ring directly. This is much
        # cheaper than re-running a circle search at every event window.
        if prior_radius is not None and self.cfg.get("predictive_refinement", True):
            local_cx, local_cy = float(prediction[0] - x0), float(prediction[1] - y0)
            local_candidates = self._search_ring_candidates_near(
                crop, local_cx, local_cy, prior_radius,
                int(self.cfg.get("predictive_candidates", 1)))
            detections = [Detection2D(timestamp_us, rcx + x0, rcy + y0, rr, score,
                                      int(round(score * 96)))
                          for rcx, rcy, rr, score in local_candidates
                          if 0.55 * prior_radius <= rr <= 1.55 * prior_radius and
                          score >= float(self.cfg.get("predictive_min_ring_score", 0.045))]
            if detections:
                return detections
            if not self.cfg.get("predictive_fallback_hough", False):
                return []
        scale = float(self.cfg.get("hough_scale", 0.5))
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 1.0)
        circles = cv2.HoughCircles(
            small, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(4.0, float(self.cfg.get("hough_min_distance_px", 20)) * scale),
            param1=float(self.cfg.get("hough_canny_threshold", 60)),
            param2=float(self.cfg.get("hough_vote_threshold", 18)),
            minRadius=max(2, round(float(self.cfg["min_radius_px"]) * scale)),
            maxRadius=max(3, round(float(self.cfg["max_radius_px"]) * scale)),
        )
        if circles is None:
            return []
        detections = []
        limit = int(self.cfg.get("maximum_candidates", 40))
        refine_limit = int(self.cfg.get("hough_refine_limit", limit))
        for candidate_index, (sx, sy, sr) in enumerate(circles[0, :limit]):
            cx, cy, radius = sx / scale, sy / scale, sr / scale
            if self.cfg.get("hough_refine_candidates", True) and candidate_index < refine_limit:
                cx, cy, radius = self._refine_circle(
                    crop, cx, cy, radius,
                    int(self.cfg.get("hough_refine_iterations", 5)))
            score = self._ring_score(crop, cx, cy, radius)
            if score < float(self.cfg.get("minimum_ring_score", 0.08)):
                continue
            detections.append(Detection2D(
                timestamp_us, cx + x0, cy + y0, radius, score,
                int(round(score * 96)),
            ))
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections
