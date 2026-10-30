"""Fit a stereo-consistent physical ball path from ambiguous RAW circle candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.config import load_config
from pingpong_event_pipeline.detector import HoughCircleBallDetector
from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae
from pingpong_event_pipeline.models import Detection2D
from pingpong_event_pipeline.offline_path import fit_smooth_stereo_candidate_path
from pingpong_event_pipeline.offline_path import SmoothStereoPath
from pingpong_event_pipeline.stereo import StereoCalibration


def collect(reader, events, timestamps, bin_us, detector, make_frames=True):
    event_times = events["timestamp"].astype(np.int64)
    candidate_bins = []; frames = []
    roi = detector.cfg.get("search_roi")
    if roi is not None:
        x0, y0, x1, y1 = map(int, roi)
        local_config = dict(detector.cfg); local_config.pop("search_roi", None)
        local_detector = HoughCircleBallDetector(local_config)
    else:
        x0 = y0 = 0; x1, y1 = reader.info.width, reader.info.height
        local_detector = detector
    for timestamp in timestamps:
        lo = np.searchsorted(event_times, timestamp - bin_us, side="right")
        hi = np.searchsorted(event_times, timestamp, side="right")
        window = events[lo:hi]
        if roi is not None and len(window):
            inside = ((window["x"] >= x0) & (window["x"] < x1) &
                      (window["y"] >= y0) & (window["y"] < y1))
            window = window[inside].copy()
            window["x"] -= x0; window["y"] -= y0
        surface = sae(window, x1 - x0, y1 - y0,
                      int(timestamp), 0.5 * bin_us)
        local = local_detector.detect(surface, int(timestamp), None)
        candidate_bins.append([
            Detection2D(value.timestamp_us, value.x + x0, value.y + y0,
                        value.radius, value.score, value.event_count) for value in local])
        if make_frames:
            if roi is None:
                frames.append(render_bgr(surface))
            else:
                full = np.zeros((2, reader.info.height, reader.info.width), np.float32)
                full[:, y0:y1, x0:x1] = surface
                frames.append(render_bgr(full))
    return candidate_bins, frames


def resample_path(path: SmoothStereoPath, timestamps_us: np.ndarray) -> SmoothStereoPath:
    """Interpolate the fitted continuous path while retaining measured-bin validity."""
    source_t = path.timestamps_us.astype(np.float64)
    target_t = np.asarray(timestamps_us, np.int64)

    def interpolate(values):
        values = np.asarray(values)
        if values.ndim == 1:
            return np.interp(target_t, source_t, values)
        return np.column_stack([
            np.interp(target_t, source_t, values[:, axis]) for axis in range(values.shape[1])])

    selected = np.full(len(target_t), -1, np.int32)
    residual = np.full(len(target_t), np.inf, np.float64)
    pair_counts = np.zeros(len(target_t), np.int32)
    locations = np.searchsorted(target_t, path.timestamps_us)
    exact = ((locations < len(target_t)) & (target_t[np.minimum(locations, len(target_t) - 1)] ==
                                             path.timestamps_us))
    locations = locations[exact]
    selected[locations] = path.selected[exact]
    residual[locations] = path.residual_m[exact]
    pair_counts[locations] = path.pair_counts[exact]
    return SmoothStereoPath(
        target_t, interpolate(path.points_m), interpolate(path.left_centers_px),
        interpolate(path.right_centers_px), interpolate(path.left_radii_px),
        interpolate(path.right_radii_px), selected, residual, pair_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--config", default="config/offline_fast_ball.yaml")
    parser.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    parser.add_argument("--start-offset-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--bin-us", type=int, default=1000)
    parser.add_argument("--detection-step-us", type=int,
                        help="Run Hough at this cadence, then resample the physical path to bin-us")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    calibration = StereoCalibration(args.calibration)
    readers = (RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw))
    common_start = max(reader.info.start_us for reader in readers)
    start = common_start + round(args.start_offset_s * 1e6)
    end = min(min(reader.info.end_us for reader in readers),
              start + round(args.duration_s * 1e6))
    detection_step_us = max(args.bin_us, args.detection_step_us or args.bin_us)
    detection_timestamps = np.arange(
        start + args.bin_us, end + 1, detection_step_us, dtype=np.int64)
    collected = []
    for reader in readers:
        events = reader.window(start, end)
        collected.append(collect(
            reader, events, detection_timestamps, args.bin_us,
            HoughCircleBallDetector(cfg["detection"]), not args.no_preview))
    candidate_sets = [value[0] for value in collected]
    frame_sets = [value[1] for value in collected]

    stereo_cfg = cfg["stereo"]
    detected_path = fit_smooth_stereo_candidate_path(
        detection_timestamps, candidate_sets[0], candidate_sets[1], calibration,
        ball_radius_m=float(stereo_cfg.get("ball_radius_m", .020)),
        max_epipolar_error_px=float(stereo_cfg.get("max_epipolar_error_px", 4.0)),
        max_reprojection_error_px=float(stereo_cfg.get("max_reprojection_error_px", 3.0)),
        maximum_radius_log_error=float(stereo_cfg.get("max_radius_scale_error", .8)),
        minimum_depth_m=float(stereo_cfg.get("minimum_depth_m", .3)),
        maximum_depth_m=float(stereo_cfg.get("maximum_depth_m", 3.0)))
    timestamps = np.arange(start + args.bin_us, end + 1, args.bin_us, dtype=np.int64)
    path = (resample_path(detected_path, timestamps)
            if detection_step_us != args.bin_us else detected_path)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    valid = (path.selected >= 0) & np.isfinite(path.residual_m)
    common_payload = dict(timestamps_us=timestamps, selected=path.selected,
                          residual_m=path.residual_m, points_m=path.points_m)
    np.savez(out / "master_path.npz", centers_px=path.left_centers_px,
             radii_px=path.left_radii_px, **common_payload)
    np.savez(out / "slave_path.npz", centers_px=path.right_centers_px,
             radii_px=path.right_radii_px, **common_payload)
    np.savez(out / "stereo_path.npz", left_centers_px=path.left_centers_px,
             right_centers_px=path.right_centers_px,
             left_radii_px=path.left_radii_px, right_radii_px=path.right_radii_px,
             pair_counts=path.pair_counts, detection_step_us=detection_step_us,
             **common_payload)

    video_path = None
    if not args.no_preview:
        video_path = out / "stereo_path.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 1e6 / args.bin_us,
                                 (readers[0].info.width * 2, readers[0].info.height))
        detected_valid = ((detected_path.selected >= 0) &
                          np.isfinite(detected_path.residual_m))
        for i, timestamp in enumerate(detection_timestamps):
            views = []
            for camera, frames in enumerate(frame_sets):
                image = frames[i]
                center = (detected_path.left_centers_px[i] if camera == 0 else
                          detected_path.right_centers_px[i])
                radius = (detected_path.left_radii_px[i] if camera == 0 else
                          detected_path.right_radii_px[i])
                color = (0, 255, 0) if detected_valid[i] else (0, 180, 255)
                cv2.circle(image, tuple(np.rint(center).astype(int)), max(2, round(radius)), color, 2)
                cv2.putText(image, f"{(timestamp-start)*1e-3:.1f} ms pairs={path.pair_counts[i]}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                views.append(image)
            writer.write(np.hstack(views))
        writer.release()

    points = path.points_m
    dt = np.gradient(timestamps).astype(np.float64) * 1e-6
    velocity = np.column_stack([np.gradient(points[:, j]) / dt for j in range(3)])
    speed = np.linalg.norm(velocity, axis=1)
    report = {
        "bins": len(timestamps),
        "detection_bins": len(detection_timestamps),
        "detection_step_us": detection_step_us,
        "bins_with_stereo_pairs": int(np.count_nonzero(path.pair_counts)),
        "path_inliers": int(np.count_nonzero(valid)),
        "median_pair_residual_mm": float(np.median(path.residual_m[valid]) * 1000),
        "p90_pair_residual_mm": float(np.percentile(path.residual_m[valid], 90) * 1000),
        "median_speed_mps": float(np.median(speed)),
        "p10_speed_mps": float(np.percentile(speed, 10)),
        "p90_speed_mps": float(np.percentile(speed, 90)),
        "video": str(video_path.resolve()) if video_path is not None else None,
    }
    (out / "path_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
