"""Render short, non-overlapping RAW event-count bins for high-speed audits."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader


def frame(events: np.ndarray, width: int, height: int, scale: int) -> np.ndarray:
    w, h = width // scale, height // scale
    counts = np.zeros((h, w), np.float32)
    if len(events):
        x = np.minimum(events["x"].astype(np.intp) // scale, w - 1)
        y = np.minimum(events["y"].astype(np.intp) // scale, h - 1)
        np.add.at(counts, (y, x), 1.0)
    positive = counts[counts > 0]
    level = float(np.percentile(positive, 99.5)) if len(positive) else 1.0
    gray = np.clip(counts * (255.0 / max(level, 1.0)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-raw", required=True)
    ap.add_argument("--slave-raw", required=True)
    ap.add_argument("--start-offset-s", type=float, required=True)
    ap.add_argument("--duration-s", type=float, default=0.2)
    ap.add_argument("--bin-us", type=int, default=1000)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    left, right = RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)
    common = max(left.info.start_us, right.info.start_us)
    start = common + round(args.start_offset_s * 1e6)
    end = start + round(args.duration_s * 1e6)
    le, re = left.window(start, end), right.window(start, end)
    size = (left.info.width // args.downsample * 2, left.info.height // args.downsample)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             1e6 / args.bin_us, size)
    for t in range(start + args.bin_us, end + 1, args.bin_us):
        lm = (le["timestamp"] > t - args.bin_us) & (le["timestamp"] <= t)
        rm = (re["timestamp"] > t - args.bin_us) & (re["timestamp"] <= t)
        view = np.hstack((frame(le[lm], left.info.width, left.info.height, args.downsample),
                          frame(re[rm], right.info.width, right.info.height, args.downsample)))
        cv2.putText(view, f"{(t-start)*1e-3:.1f} ms", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(view)
    writer.release()
    print({"output": str(output.resolve()), "frames": round((end-start)/args.bin_us),
           "master_events": len(le), "slave_events": len(re)})


if __name__ == "__main__":
    main()
