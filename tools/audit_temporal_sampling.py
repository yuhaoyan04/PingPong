"""Audit timestamp resolution and event support at several ball-ROI windows.

This tool deliberately does not estimate spin.  It answers the earlier question:
does a recorded ball segment contain enough temporally resolved events to justify
trying a dot/orientation estimator at a requested update interval?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_event_pipeline.events import RawWindowReader


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def camera_audit(
    raw_path: Path,
    timestamps_us: np.ndarray,
    centers_px: np.ndarray,
    radii_px: np.ndarray,
    windows_us: list[int],
    roi_scale: float,
) -> dict:
    reader = RawWindowReader(raw_path)
    events = reader.window(
        int(timestamps_us[0] - max(windows_us)), int(timestamps_us[-1] + 1)
    )
    event_times = events["timestamp"].astype(np.int64, copy=False)
    negative_jumps = int(np.count_nonzero(np.diff(event_times) < 0))
    if negative_jumps:
        # Some SDK/raw versions may return packet blocks out of order.  Window
        # lookup below uses searchsorted, so make that precondition explicit.
        order = np.argsort(event_times, kind="stable")
        events = events[order]
        event_times = events["timestamp"].astype(np.int64, copy=False)
    unique_times, events_per_time = np.unique(event_times, return_counts=True)
    positive_steps = np.diff(unique_times)
    positive_steps = positive_steps[positive_steps > 0]

    result = {
        "raw": str(raw_path.resolve()),
        "events_loaded": int(len(events)),
        "timestamp": {
            "negative_adjacent_jumps_before_sort": negative_jumps,
            "sorted_for_window_statistics": bool(negative_jumps),
            "smallest_positive_step_us": int(np.min(positive_steps)),
            "median_positive_step_us": float(np.median(positive_steps)),
            "occupied_microseconds_fraction": float(
                len(unique_times) / (timestamps_us[-1] - timestamps_us[0] + 1)
            ),
            "events_per_occupied_timestamp": distribution(events_per_time),
        },
        "windows": {},
    }

    for window_us in windows_us:
        counts: list[int] = []
        active_pixels: list[int] = []
        for timestamp, center, radius in zip(timestamps_us, centers_px, radii_px):
            lo = np.searchsorted(event_times, timestamp - window_us, side="left")
            hi = np.searchsorted(event_times, timestamp, side="right")
            current = events[lo:hi]
            dx = current["x"].astype(np.float64) - float(center[0])
            dy = current["y"].astype(np.float64) - float(center[1])
            inside = dx * dx + dy * dy <= (float(radius) * roi_scale) ** 2
            roi_events = current[inside]
            counts.append(int(len(roi_events)))
            if len(roi_events):
                packed = (
                    roi_events["y"].astype(np.int64) * reader.info.width
                    + roi_events["x"].astype(np.int64)
                )
                active_pixels.append(int(len(np.unique(packed))))
            else:
                active_pixels.append(0)
        count_array = np.asarray(counts)
        result["windows"][str(window_us)] = {
            "event_count": distribution(count_array),
            "active_pixels": distribution(np.asarray(active_pixels)),
            "nonempty_fraction": float(np.mean(count_array > 0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit temporal/event support in a fitted ball segment"
    )
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--stereo-path", required=True)
    parser.add_argument("--first-index", type=int, required=True)
    parser.add_argument("--last-index", type=int, required=True)
    parser.add_argument(
        "--windows-us", type=int, nargs="+", default=[250, 500, 1000, 2500, 5000]
    )
    parser.add_argument("--roi-scale", type=float, default=1.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    path = np.load(args.stereo_path)
    indices = np.arange(args.first_index, args.last_index + 1)
    if len(indices) < 2 or indices[0] < 0 or indices[-1] >= len(path["timestamps_us"]):
        raise ValueError("Invalid path index range")
    windows_us = sorted(set(int(value) for value in args.windows_us))
    if not windows_us or windows_us[0] <= 0:
        raise ValueError("All temporal windows must be positive")
    timestamps = path["timestamps_us"][indices].astype(np.int64)

    report = {
        "purpose": "sampling_feasibility_only_not_a_spin_measurement",
        "segment": {
            "first_index": int(indices[0]),
            "last_index": int(indices[-1]),
            "duration_us": int(timestamps[-1] - timestamps[0]),
            "path_samples": int(len(indices)),
            "path_step_us": float(np.median(np.diff(timestamps))),
        },
        "roi_scale": float(args.roi_scale),
        "master": camera_audit(
            Path(args.master_raw), timestamps, path["left_centers_px"][indices],
            path["left_radii_px"][indices], windows_us, args.roi_scale
        ),
        "slave": camera_audit(
            Path(args.slave_raw), timestamps, path["right_centers_px"][indices],
            path["right_radii_px"][indices], windows_us, args.roi_scale
        ),
        "interpretation": [
            "A nonempty ROI is necessary but does not prove that individual dots are separable.",
            "Shorter windows reduce angular motion but also reduce dot evidence.",
            "The next experiment must visualize dot candidates at every requested window.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
