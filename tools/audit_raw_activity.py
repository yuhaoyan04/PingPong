"""Render logarithmic event-count sheets directly from a synchronized RAW pair."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader


def count_image(events: np.ndarray, width: int, height: int) -> np.ndarray:
    count = np.zeros((height, width), np.float32)
    if len(events):
        x = events["x"].astype(np.intp); y = events["y"].astype(np.intp)
        valid = (x < width) & (y < height)
        np.add.at(count, (y[valid], x[valid]), 1)
    scaled = np.log1p(count)
    high = np.percentile(scaled[scaled > 0], 99.5) if np.any(scaled > 0) else 1.0
    gray = np.clip(scaled * 255 / max(float(high), 1.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def make_sheet(items: list[tuple[float, np.ndarray, np.ndarray, int, int]], columns: int = 3) -> np.ndarray:
    tiles = []
    for t, left, right, nl, nr in items:
        frame = np.hstack((left, right))
        frame = cv2.resize(frame, (1280, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"t={t:.2f}s  M={nl} S={nr}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank.copy())
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-raw", required=True)
    ap.add_argument("--slave-raw", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window-us", type=int, default=100000)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()
    left = RawWindowReader(args.master_raw); right = RawWindowReader(args.slave_raw)
    origin = max(left.info.start_us, right.info.start_us)
    end = min(left.info.end_us, right.info.end_us)
    left_events = left.window(origin, end); right_events = right.window(origin, end)
    left_times = left_events["timestamp"].astype(np.int64)
    right_times = right_events["timestamp"].astype(np.int64)
    edges = np.arange(origin, end + args.window_us, args.window_us, dtype=np.int64)
    left_count, _ = np.histogram(left_times, edges)
    right_count, _ = np.histogram(right_times, edges)
    ranked = np.argsort(left_count + right_count)[::-1]
    gap = max(1, round(500000 / args.window_us)); selected: list[int] = []
    for i in ranked:
        if all(abs(int(i) - old) >= gap for old in selected):
            selected.append(int(i))
        if len(selected) >= args.samples:
            break
    selected.sort()
    items = []
    for i in selected:
        lo, hi = int(edges[i]), int(edges[i + 1])
        lli = np.searchsorted(left_times, lo); lhi = np.searchsorted(left_times, hi)
        rli = np.searchsorted(right_times, lo); rhi = np.searchsorted(right_times, hi)
        le, re = left_events[lli:lhi], right_events[rli:rhi]
        t = (hi - origin) * 1e-6
        items.append((t, count_image(le, 1280, 720), count_image(re, 1280, 720), len(le), len(re)))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), make_sheet(items))
    print({"windows": len(edges) - 1,
           "selected_s": [round((edges[i + 1] - origin) * 1e-6, 3) for i in selected],
           "output": str(out.resolve())})


if __name__ == "__main__":
    main()
