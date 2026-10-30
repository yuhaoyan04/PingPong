"""Track a high-speed ball from its complementary event-polarity crescents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.crescent import CrescentBallDetector
from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae


def smooth(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples <= 1:
        return values.copy()
    if samples % 2 == 0:
        samples += 1
    result = np.empty_like(values)
    for axis in range(values.shape[1]):
        padded = np.pad(values[:, axis], (samples // 2, samples // 2), mode="edge")
        result[:, axis] = np.convolve(padded, np.ones(samples) / samples, mode="valid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ON/OFF 新月高速球轨迹；不计算转速")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--origin-us", type=int, required=True)
    parser.add_argument("--start-offset-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--window-us", type=int, default=1000)
    parser.add_argument("--step-us", type=int, default=1000)
    parser.add_argument("--search-roi", type=int, nargs=4, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--min-radius", type=float, default=15)
    parser.add_argument("--max-radius", type=float, default=140)
    parser.add_argument("--sigma-px", type=float, default=4)
    parser.add_argument("--threshold-fraction", type=float, default=.16)
    parser.add_argument("--minimum-lobe-events", type=int, default=30)
    parser.add_argument("--max-center-step", type=float, default=45)
    parser.add_argument("--smooth-samples", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    reader = RawWindowReader(args.raw)
    start = args.origin_us + round(args.start_offset_s * 1e6)
    end = min(reader.info.end_us, start + round(args.duration_s * 1e6))
    events = reader.window(start, end)
    event_times = events["timestamp"].astype(np.int64, copy=False)
    if np.any(np.diff(event_times) < 0):
        events = events[np.argsort(event_times, kind="stable")]
        event_times = events["timestamp"].astype(np.int64, copy=False)
    detector = CrescentBallDetector(
        tuple(args.search_roi), args.min_radius, args.max_radius, args.sigma_px,
        args.threshold_fraction, args.minimum_lobe_events,
        maximum_lobe_separation_px=2.2 * args.max_radius)
    timestamps = np.arange(start + args.window_us, end + 1, args.step_us, dtype=np.int64)
    measured = np.full((len(timestamps), 3), np.nan, np.float64)
    scores = np.zeros(len(timestamps), np.float64)
    supports = np.zeros(len(timestamps), np.int32)
    previous = None; velocity = np.zeros(2, np.float64)
    all_candidates: list[list] = []
    for index, timestamp in enumerate(timestamps):
        lo = np.searchsorted(event_times, timestamp - args.window_us, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        predicted = None if previous is None else np.r_[previous[:2] + velocity, previous[2]]
        candidates = detector.detect(events[lo:hi], int(timestamp), predicted)
        all_candidates.append(candidates)
        if previous is None:
            chosen = candidates[0] if candidates else None
        else:
            plausible = [value for value in candidates
                         if np.linalg.norm(value.point - predicted[:2]) <= args.max_center_step
                         and .55 * previous[2] <= value.radius <= 1.55 * previous[2]]
            chosen = max(plausible, key=lambda value: value.score, default=None)
        if chosen is None:
            if previous is not None:
                measured[index] = np.r_[previous[:2] + velocity, previous[2]]
                previous = measured[index].copy()
            continue
        current = np.array([chosen.x, chosen.y, chosen.radius], np.float64)
        if previous is not None:
            velocity = .65 * velocity + .35 * (current[:2] - previous[:2])
        measured[index] = current; scores[index] = chosen.score; supports[index] = chosen.event_count
        previous = current

    valid = np.isfinite(measured).all(axis=1) & (scores > 0)
    if np.count_nonzero(valid) < 4:
        raise RuntimeError("有效极性新月检测少于 4 帧")
    # Restrict interpolation to the measured flight interval; predictions outside it are invalid.
    first, last = np.flatnonzero(valid)[[0, -1]]
    for axis in range(3):
        measured[first:last + 1, axis] = np.interp(
            timestamps[first:last + 1], timestamps[valid], measured[valid, axis])
    path = smooth(measured[first:last + 1], args.smooth_samples)
    path_timestamps = timestamps[first:last + 1]
    np.savez(output / "monocular_path.npz", timestamps_us=path_timestamps,
             centers_px=path[:, :2], radii_px=path[:, 2], scores=scores[first:last + 1],
             supports=supports[first:last + 1], measured_circles_px=measured[first:last + 1])

    x0, y0, x1, y1 = args.search_roi
    sample_indices = set(np.linspace(first, last, min(16, last - first + 1), dtype=int))
    samples = []
    for index in sorted(sample_indices):
        timestamp = timestamps[index]
        lo = np.searchsorted(event_times, timestamp - args.window_us, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        surface = sae(events[lo:hi], reader.info.width, reader.info.height,
                      int(timestamp), .5 * args.window_us)
        image = render_bgr(surface)[y0:y1, x0:x1].copy()
        circle = path[index - first]
        center = tuple(np.rint(circle[:2] - [x0, y0]).astype(int))
        color = (0, 255, 0) if scores[index] > 0 else (0, 180, 255)
        cv2.circle(image, center, max(2, round(circle[2])), color, 2)
        cv2.putText(image, f"{(timestamp-start)/1000:.0f} ms score={scores[index]:.2f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        samples.append(image)
    if samples:
        columns = 4; blank = np.zeros_like(samples[0]); rows = []
        for index in range(0, len(samples), columns):
            chunk = samples[index:index + columns]
            rows.append(np.hstack(chunk + [blank] * (columns - len(chunk))))
        cv2.imwrite(str(output / "crescent_track_contact_sheet.jpg"), np.vstack(rows))

    report = {
        "purpose": "polarity_crescent_ball_trajectory_no_spin",
        "timestamps": len(path_timestamps),
        "duration_ms": float((path_timestamps[-1] - path_timestamps[0]) / 1000),
        "measured_fraction": float(np.mean(scores[first:last + 1] > 0)),
        "measured_frames": int(np.count_nonzero(scores[first:last + 1] > 0)),
        "median_score": float(np.median(scores[scores > 0])),
        "median_support_events": float(np.median(supports[supports > 0])),
        "radius_px_median": float(np.median(path[:, 2])),
        "center_bounds_px": [float(path[:, 0].min()), float(path[:, 1].min()),
                             float(path[:, 0].max()), float(path[:, 1].max())],
        "path": str((output / "monocular_path.npz").resolve()),
        "contact_sheet": str((output / "crescent_track_contact_sheet.jpg").resolve()),
    }
    (output / "crescent_track_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
