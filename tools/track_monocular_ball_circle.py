"""Track the image-space ball circle in a monocular research clip.

This tool deliberately produces no depth, physical trajectory, or spin result.  Its
only job is to make later marker observations comparable in a ball-centred frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from pingpong_event_pipeline.detector import HoughCircleBallDetector
from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae


def smooth(values: np.ndarray, samples: int) -> np.ndarray:
    window = min(samples if samples % 2 else samples - 1, len(values) // 2 * 2 - 1)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=min(2, window - 1), axis=0)


def resolve_start_us(reader_start_us: int, start_offset_s: float,
                     origin_us: int | None = None) -> int:
    """Resolve an offset against an explicit shared origin when provided."""
    return int(reader_start_us if origin_us is None else origin_us) + round(
        start_offset_s * 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单目球轮廓轨迹，仅用于球心补偿；不计算三维位置或转速")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--start-offset-s", type=float, required=True)
    parser.add_argument("--origin-us", type=int,
                        help="shared absolute timestamp origin for synchronized RAW pairs")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--window-us", type=int, default=5000)
    parser.add_argument("--detection-step-us", type=int, default=5000)
    parser.add_argument("--output-step-us", type=int, default=1000)
    parser.add_argument("--search-roi", type=int, nargs=4, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--min-radius", type=int, required=True)
    parser.add_argument("--max-radius", type=int, required=True)
    parser.add_argument("--max-center-step", type=float, default=35.0,
                        help="maximum accepted displacement per detection step")
    parser.add_argument("--smooth-samples", type=int, default=9)
    parser.add_argument("--display-fps", type=float, default=25.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reader = RawWindowReader(args.raw)
    start = resolve_start_us(reader.info.start_us, args.start_offset_s, args.origin_us)
    if start < reader.info.start_us:
        raise ValueError("resolved start precedes this RAW recording")
    end = min(reader.info.end_us, start + round(args.duration_s * 1e6))
    events = reader.window(start, end)
    event_times = events["timestamp"].astype(np.int64, copy=False)
    if np.any(np.diff(event_times) < 0):
        events = events[np.argsort(event_times, kind="stable")]
        event_times = events["timestamp"].astype(np.int64, copy=False)

    cfg = {
        "search_roi": args.search_roi,
        "min_radius_px": args.min_radius,
        "max_radius_px": args.max_radius,
        "hough_scale": .5,
        "hough_min_distance_px": max(30, args.min_radius * .65),
        "hough_canny_threshold": 35,
        "hough_vote_threshold": 12,
        "maximum_candidates": 20,
        "hough_refine_limit": 20,
        "hough_refine_candidates": True,
        "minimum_ring_score": .04,
    }
    detector = HoughCircleBallDetector(cfg)
    detection_timestamps = np.arange(
        start + args.window_us, end + 1, args.detection_step_us, dtype=np.int64)
    measured = []
    scores = []
    rejected_bins = 0
    previous = None
    velocity = np.zeros(2, np.float64)
    for timestamp in detection_timestamps:
        lo = np.searchsorted(event_times, timestamp - args.window_us, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        surface = sae(events[lo:hi], reader.info.width, reader.info.height,
                      int(timestamp), .5 * args.window_us)
        candidates = detector.detect(surface, int(timestamp))
        if previous is None:
            chosen = candidates[0] if candidates else None
        else:
            predicted = previous[:2] + velocity
            plausible = [candidate for candidate in candidates
                         if np.linalg.norm(np.array([candidate.x, candidate.y]) - predicted)
                         <= args.max_center_step
                         and abs(candidate.radius - previous[2]) <= 0.20 * previous[2]]
            chosen = max(
                plausible,
                key=lambda candidate: candidate.score
                - .004 * np.linalg.norm(np.array([candidate.x, candidate.y]) - predicted)
                - .006 * abs(candidate.radius - previous[2]),
                default=None,
            )
        if chosen is None:
            rejected_bins += 1
            if previous is None:
                measured.append([np.nan, np.nan, np.nan])
                scores.append(0.0)
                continue
            current = previous.copy()
            current[:2] += velocity
            measured.append(current)
            scores.append(0.0)
            previous = current
            continue
        current = np.array([chosen.x, chosen.y, chosen.radius], np.float64)
        if previous is not None:
            velocity = .65 * velocity + .35 * (current[:2] - previous[:2])
        measured.append(current)
        scores.append(chosen.score)
        previous = current

    measured = np.asarray(measured, np.float64)
    scores = np.asarray(scores, np.float64)
    valid = np.isfinite(measured).all(axis=1)
    if np.count_nonzero(valid) < 4:
        raise RuntimeError("有效球轮廓检测少于 4 帧，无法形成轨迹")
    for axis in range(3):
        measured[:, axis] = np.interp(
            detection_timestamps, detection_timestamps[valid], measured[valid, axis])
    smoothed = smooth(measured, args.smooth_samples)
    timestamps = np.arange(start + args.window_us, end + 1,
                           args.output_step_us, dtype=np.int64)
    path = np.column_stack([
        np.interp(timestamps, detection_timestamps, smoothed[:, axis])
        for axis in range(3)
    ])
    score_path = np.interp(timestamps, detection_timestamps, scores)
    np.savez(
        output / "monocular_path.npz",
        timestamps_us=timestamps,
        centers_px=path[:, :2],
        radii_px=path[:, 2],
        scores=score_path,
        detection_timestamps_us=detection_timestamps,
        measured_circles_px=measured,
        measured_scores=scores,
    )

    x0, y0, x1, y1 = args.search_roi
    preview_path = output / "circle_track.mp4"
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, (x1 - x0, y1 - y0))
    samples = []
    sample_indices = set(np.linspace(0, len(detection_timestamps) - 1,
                                     min(12, len(detection_timestamps)), dtype=int))
    for index, timestamp in enumerate(detection_timestamps):
        lo = np.searchsorted(event_times, timestamp - args.window_us, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        surface = sae(events[lo:hi], reader.info.width, reader.info.height,
                      int(timestamp), .5 * args.window_us)
        frame = render_bgr(surface)[y0:y1, x0:x1].copy()
        circle = smoothed[index]
        center = tuple(np.rint(circle[:2] - [x0, y0]).astype(int))
        color = (0, 255, 0) if scores[index] > 0 else (0, 180, 255)
        cv2.circle(frame, center, round(circle[2]), color, 2)
        cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(frame, f"{(timestamp-start)/1000:.0f} ms score={scores[index]:.3f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        writer.write(frame)
        if index in sample_indices:
            samples.append(frame)
    writer.release()
    if samples:
        columns = 4
        rows = []
        blank = np.zeros_like(samples[0])
        for i in range(0, len(samples), columns):
            rows.append(np.hstack(samples[i:i + columns]
                                  + [blank] * (columns - len(samples[i:i + columns]))))
        cv2.imwrite(str(output / "circle_track_contact_sheet.jpg"), np.vstack(rows))

    residual = measured - smoothed
    report = {
        "purpose": "monocular_image_circle_compensation_only_no_depth_or_spin",
        "raw": str(Path(args.raw).resolve()),
        "start_offset_s": args.start_offset_s,
        "origin_us": args.origin_us if args.origin_us is not None else reader.info.start_us,
        "start_timestamp_us": int(start),
        "duration_s": (end - start) * 1e-6,
        "window_us": args.window_us,
        "detection_step_us": args.detection_step_us,
        "output_step_us": args.output_step_us,
        "detection_bins": len(detection_timestamps),
        "bins_without_accepted_measurement": rejected_bins,
        "accepted_fraction": float(np.mean(scores > 0)),
        "median_ring_score": float(np.median(scores[scores > 0])),
        "center_fit_residual_median_px": float(np.median(np.linalg.norm(residual[:, :2], axis=1))),
        "center_fit_residual_p90_px": float(np.percentile(np.linalg.norm(residual[:, :2], axis=1), 90)),
        "radius_range_px": [float(path[:, 2].min()), float(path[:, 2].max())],
        "center_bounds_px": [float(path[:, 0].min()), float(path[:, 1].min()),
                             float(path[:, 0].max()), float(path[:, 1].max())],
        "path": str((output / "monocular_path.npz").resolve()),
        "preview": str(preview_path.resolve()),
        "contact_sheet": str((output / "circle_track_contact_sheet.jpg").resolve()),
    }
    (output / "circle_track_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
