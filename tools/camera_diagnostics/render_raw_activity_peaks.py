"""Render strong activity windows from a RAW file without loading all events."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def open_reader(path):
    from dvsense_driver.raw_file_reader import RawFileReader

    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"不能读取 {path}")
    return reader


def scan_activity(path, window_us):
    reader = open_reader(path)
    ok_start, start = reader.get_start_timestamp()
    ok_end, end = reader.get_end_timestamp()
    if not ok_start or not ok_end:
        raise RuntimeError("RAW 时间范围不可用")
    counts = np.zeros((end - start) // window_us + 2, dtype=np.int64)
    while not reader.reached_end_of_events():
        events = reader.get_n_time_events(20_000)
        if events.size:
            indices = (events["timestamp"].astype(np.int64) - start) // window_us
            indices = indices[(indices >= 0) & (indices < len(counts))]
            counts += np.bincount(indices, minlength=len(counts))
        if reader.get_current_pos_timestamp() + 20_000 >= end:
            break
    return start, end, counts, reader.get_width(), reader.get_height()


def render_windows(path, centers, window_us, width, height):
    reader = open_reader(path)
    images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in centers]
    totals = [0] * len(centers)
    while not reader.reached_end_of_events():
        events = reader.get_n_time_events(20_000)
        if events.size:
            t = events["timestamp"].astype(np.int64)
            for index, center in enumerate(centers):
                selected = events[(t >= center - window_us // 2) & (t < center + window_us // 2)]
                if not selected.size:
                    continue
                x = selected["x"].astype(np.int64)
                y = selected["y"].astype(np.int64)
                p = selected["polarity"].astype(bool)
                images[index][y[~p], x[~p]] = (50, 80, 255)
                images[index][y[p], x[p]] = (255, 240, 40)
                totals[index] += len(selected)
    return images, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-us", type=int, default=20_000)
    parser.add_argument("--peaks", type=int, default=8)
    args = parser.parse_args()

    start, end, counts, width, height = scan_activity(args.raw, args.window_us)
    chosen = []
    for index in np.argsort(counts)[::-1]:
        center = start + int(index) * args.window_us + args.window_us // 2
        if center < start + 1_000_000 or center > end - 1_000_000:
            continue
        if all(abs(center - old) >= 3 * args.window_us for old in chosen):
            chosen.append(center)
        if len(chosen) == args.peaks:
            break

    images, totals = render_windows(args.raw, chosen, args.window_us, width, height)
    panels = []
    for center, event_image, total in sorted(zip(chosen, images, totals)):
        panel = Image.fromarray(event_image).resize((640, 360))
        ImageDraw.Draw(panel).text(
            (15, 15), f"t={center / 1e6:.3f}s  n={total}", fill=(255, 255, 255)
        )
        panels.append(np.asarray(panel))
    rows = [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
    if len(rows[-1]) != len(rows[0]):
        rows[-1] = np.hstack([rows[-1], np.zeros_like(panels[0])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
