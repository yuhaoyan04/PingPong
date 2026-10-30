"""Extract reviewable stereo-activity clip candidates from a long RAW pair."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.config import load_config
from pingpong_event_pipeline.events import RawWindowReader


FIELDS = [
    "clip_id", "run_id", "start_offset_s", "end_offset_s", "peak_offset_s",
    "duration_s", "master_peak_events", "slave_peak_events", "joint_score",
    "label", "review_status", "notes",
]


def count_roi(reader: RawWindowReader, edges: np.ndarray, roi: tuple[int, int, int, int]):
    x0, y0, x1, y1 = roi
    counts = np.zeros(len(edges) - 1, np.int64)
    for first in range(0, len(counts), 500):
        last = min(first + 500, len(counts))
        events = reader.window(int(edges[first]), int(edges[last]))
        inside = (
            (events["x"] >= x0) & (events["x"] < x1)
            & (events["y"] >= y0) & (events["y"] < y1)
        )
        counts[first:last] = np.histogram(
            events["timestamp"][inside].astype(np.int64), edges[first:last + 1]
        )[0]
    return counts


def select_peaks(score: np.ndarray, edges: np.ndarray, count: int, separation_us: int):
    selected: list[int] = []
    for index in np.argsort(score)[::-1]:
        center = (int(edges[index]) + int(edges[index + 1])) // 2
        if score[index] <= 0:
            continue
        if any(abs(center - old) < separation_us for old in selected):
            continue
        selected.append(center)
        if len(selected) >= count:
            break
    return sorted(selected)


def event_image(events: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    image = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    inside = (
        (events["x"] >= x0) & (events["x"] < x1)
        & (events["y"] >= y0) & (events["y"] < y1)
    )
    current = events[inside]
    if len(current):
        x = current["x"].astype(np.intp) - x0
        y = current["y"].astype(np.intp) - y0
        p = current["polarity"].astype(bool)
        image[y[~p], x[~p], 0] = 255
        image[y[p], x[p], 1:] = 255
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="提取双路共同活动片段，供人工分类")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/realtime_trajectory.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--bin-us", type=int, default=20_000)
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--separation-ms", type=float, default=500)
    parser.add_argument("--pre-ms", type=float, default=40)
    parser.add_argument("--post-ms", type=float, default=300)
    parser.add_argument("--overview-ms", type=float, default=20)
    parser.add_argument("--known-serve-offset-s", type=float)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    master_path = Path(args.master_raw)
    slave_path = Path(args.slave_raw)
    run_id = args.run_id or master_path.parent.name
    readers = [RawWindowReader(master_path), RawWindowReader(slave_path)]
    origin = max(reader.info.start_us for reader in readers)
    end = min(reader.info.end_us for reader in readers)
    edges = np.arange(origin, end + args.bin_us, args.bin_us, dtype=np.int64)
    config = load_config(args.config)
    roi = tuple(map(int, config["detection"]["search_roi"]))
    counts = [count_roi(reader, edges, roi) for reader in readers]
    score = np.sqrt(counts[0].astype(np.float64) * counts[1].astype(np.float64))
    centers = select_peaks(
        score, edges, max(1, args.clips), round(args.separation_ms * 1000)
    )

    pre_us = round(args.pre_ms * 1000)
    post_us = round(args.post_ms * 1000)
    overview_us = round(args.overview_ms * 1000)
    rows = []
    overview_tiles = []
    for number, center in enumerate(centers, 1):
        index = int(np.searchsorted(edges, center, side="right") - 1)
        start = max(origin, center - pre_us)
        finish = min(end, center + post_us)
        peak_offset = (center - origin) * 1e-6
        known = (
            args.known_serve_offset_s is not None
            and abs(peak_offset - args.known_serve_offset_s) <= max(.1, args.bin_us * 1e-6)
        )
        if known:
            # The supplied offset is the already reviewed clip start, which is
            # more precise than the centre of a coarse activity bin.
            start = max(origin, origin + round(args.known_serve_offset_s * 1e6))
            finish = min(end, start + pre_us + post_us)
        row = {
            "clip_id": f"{run_id}_c{number:02d}",
            "run_id": run_id,
            "start_offset_s": round((start - origin) * 1e-6, 6),
            "end_offset_s": round((finish - origin) * 1e-6, 6),
            "peak_offset_s": round(peak_offset, 6),
            "duration_s": round((finish - start) * 1e-6, 6),
            "master_peak_events": int(counts[0][index]),
            "slave_peak_events": int(counts[1][index]),
            "joint_score": round(float(score[index]), 3),
            "label": "serve" if known else "unknown_activity",
            "review_status": "existing_path_verified" if known else "unreviewed",
            "notes": "Known reviewed clip with an existing fitted trajectory" if known else "",
        }
        rows.append(row)
        camera_images = []
        for reader in readers:
            events = reader.window(center - overview_us // 2, center + overview_us // 2)
            camera_images.append(event_image(events, roi))
        tile = np.hstack(camera_images)
        tile = cv2.resize(tile, (1320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile, f"{row['clip_id']} peak={peak_offset:.3f}s score={score[index]:.0f}",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA
        )
        overview_tiles.append(tile)

    csv_path = output / "clip_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "run_id": run_id,
        "master_raw": str(master_path.resolve()),
        "slave_raw": str(slave_path.resolve()),
        "origin_us": int(origin),
        "common_duration_s": (end - origin) * 1e-6,
        "search_roi": list(roi),
        "bin_us": args.bin_us,
        "selection": "largest separated geometric-mean stereo activity peaks",
        "warning": "Activity peaks are candidates, not automatic ball/serve labels.",
        "clips": rows,
    }
    (output / "clip_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if overview_tiles:
        cv2.imwrite(str(output / "clip_overview.jpg"), np.vstack(overview_tiles))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
