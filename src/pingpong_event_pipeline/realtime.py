from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .detector import HoughCircleBallDetector
from .events import IncrementalTimeSurface
from .models import Track2D
from .pipeline import _detect_with_normalized_roi, _minimum_roi_event_count
from .stereo import StereoCalibration, select_stereo_detection_pair
from .tracking import BallTracker2D
from .trajectory import TrajectoryFilter3D


@dataclass
class CausalStereoResult:
    timestamp_us: int
    master_track: Optional[Track2D]
    slave_track: Optional[Track2D]
    measured_position_m: Optional[np.ndarray]
    filtered_position_m: Optional[np.ndarray]
    velocity_mps: Optional[np.ndarray]
    reprojection_error_px: Optional[float]

    @property
    def speed_mps(self) -> Optional[float]:
        return None if self.velocity_mps is None else float(np.linalg.norm(self.velocity_mps))


class CausalStereoTracker:
    """Stateful causal trajectory core shared by live SDK and replay runtimes."""

    def __init__(self, config: dict, calibration_path: str | Path):
        self.config = config
        self.event_cfg = config["events"]; self.detector_cfg = config["detection"]
        width, height = config["cameras"]["width"], config["cameras"]["height"]
        self.width, self.height = int(width), int(height)
        self.calibration = StereoCalibration(calibration_path)
        self.master_detector = HoughCircleBallDetector(self.detector_cfg)
        self.slave_detector = HoughCircleBallDetector(self.detector_cfg)
        missed = int(self.detector_cfg["max_missed_windows"])
        self.master_tracker = BallTracker2D(missed); self.slave_tracker = BallTracker2D(missed)
        self.master_accumulator = IncrementalTimeSurface(width, height)
        self.slave_accumulator = IncrementalTimeSurface(width, height)
        self.trajectory = TrajectoryFilter3D(config["trajectory"])
        self.index = 0; self.previous_point = None; self.previous_timestamp = None
        self.consecutive_stereo_measurements = 0

    def process(self, timestamp_us: int, master_events, slave_events) -> CausalStereoResult:
        self.master_accumulator.ingest(master_events); self.slave_accumulator.ingest(slave_events)
        mt, st = self.master_tracker, self.slave_tracker
        mp = mt.x[[0, 1, 4]] if mt.initialized else None
        sp = st.x[[0, 1, 4]] if st.initialized else None
        base_interval = int(self.detector_cfg.get("interval_us", self.event_cfg["step_us"]))
        settled_after = int(self.detector_cfg.get("settled_after_stereo_measurements", 0))
        interval = (int(self.detector_cfg.get("settled_interval_us", base_interval))
                    if settled_after > 0 and self.consecutive_stereo_measurements >= settled_after
                    else base_interval)
        stride = max(1, round(interval / self.event_cfg["step_us"]))
        initialized = mt.initialized and st.initialized
        if initialized:
            run_detection = self.index % stride == 0
        else:
            bootstrap_interval = int(self.detector_cfg.get("bootstrap_interval_us", interval))
            bootstrap_stride = max(1, round(bootstrap_interval / self.event_cfg["step_us"]))
            activity = _minimum_roi_event_count(master_events, slave_events, self.detector_cfg)
            burst = activity >= int(self.detector_cfg.get(
                "bootstrap_burst_roi_events", 2**31 - 1))
            run_detection = ((burst or self.index % bootstrap_stride == 0) and
                             activity >= int(self.detector_cfg.get("bootstrap_min_roi_events", 0)))
        if run_detection:
            master_candidates, _ = _detect_with_normalized_roi(
                self.master_accumulator, timestamp_us, mt, self.master_detector, mp,
                self.event_cfg, self.config["spin"], self.width, self.height)
            slave_candidates, _ = _detect_with_normalized_roi(
                self.slave_accumulator, timestamp_us, st, self.slave_detector, sp,
                self.event_cfg, self.config["spin"], self.width, self.height)
            if not mt.initialized or not st.initialized:
                score = float(self.detector_cfg.get("bootstrap_min_score", 0.0))
                radius = float(self.detector_cfg.get("bootstrap_min_radius_px", 0.0))
                master_candidates = [d for d in master_candidates if d.score >= score and d.radius >= radius]
                slave_candidates = [d for d in slave_candidates if d.score >= score and d.radius >= radius]
        else:
            master_candidates = []; slave_candidates = []
        stereo_cfg = self.config["stereo"]
        previous = self.previous_point; jump_limit = None
        if self.previous_timestamp is not None:
            gap_s = (timestamp_us - self.previous_timestamp) * 1e-6
            if gap_s > stereo_cfg.get("continuity_reset_s", .05):
                previous = None
            else:
                jump_limit = (stereo_cfg.get("maximum_ball_speed_mps", 30.0) * gap_s +
                              stereo_cfg.get("position_jump_allowance_m", .02))
        pair = select_stereo_detection_pair(
            master_candidates, slave_candidates, self.calibration,
            (stereo_cfg["max_epipolar_error_px"] if initialized else
             self.detector_cfg.get("bootstrap_max_epipolar_error_px",
                                   stereo_cfg["max_epipolar_error_px"])),
            stereo_cfg.get("max_radius_ratio", 2.0),
            stereo_cfg.get("ball_radius_m"), stereo_cfg.get("minimum_depth_m", .3),
            stereo_cfg.get("maximum_depth_m", 3.0), stereo_cfg.get("max_radius_scale_error", .8),
            mp[:2] if mp is not None else None, sp[:2] if sp is not None else None,
            stereo_cfg.get("maximum_prediction_error_px", 60.0), previous, jump_limit)
        if run_detection:
            master_track = mt.update(timestamp_us, [pair[0]] if pair else [])
            slave_track = st.update(timestamp_us, [pair[1]] if pair else [])
        else:
            master_track = mt.advance(timestamp_us); slave_track = st.advance(timestamp_us)
        measured = filtered = velocity = None; reprojection = None
        if pair:
            point, error = self.calibration.triangulate(pair[0].point, pair[1].point)
            if np.isfinite(point).all() and error <= stereo_cfg["max_reprojection_error_px"]:
                self.previous_point = point.copy(); self.previous_timestamp = timestamp_us
                filtered, velocity = self.trajectory.update(timestamp_us, point)
                measured, reprojection = point, float(error)
                self.consecutive_stereo_measurements += 1
            elif run_detection:
                self.consecutive_stereo_measurements = max(0, self.consecutive_stereo_measurements - 2)
        elif run_detection:
            self.consecutive_stereo_measurements = max(0, self.consecutive_stereo_measurements - 2)
        self.index += 1
        return CausalStereoResult(timestamp_us, master_track, slave_track, measured,
                                  filtered, velocity, reprojection)
