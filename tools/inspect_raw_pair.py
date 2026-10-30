"""把已同步 RAW 渲染成双路预览，用于离线检查视野、事件率和时间覆盖。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae, synchronized_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-raw", required=True)
    ap.add_argument("--slave-raw", required=True)
    ap.add_argument("--output", default="output/raw_pair_preview.mp4")
    ap.add_argument("--window-us", type=int, default=10000)
    ap.add_argument("--step-us", type=int, default=10000)
    ap.add_argument("--max-frames", type=int, default=1000)
    args = ap.parse_args()
    left, right = RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    scale = 0.5
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             1_000_000 / args.step_us, (1280, 360))
    counts = [0, 0]; frames = 0
    for ts, le, re in synchronized_windows(left, right, args.window_us, args.step_us):
        if frames >= args.max_frames: break
        counts[0] += len(le); counts[1] += len(re)
        li = render_bgr(sae(le, 1280, 720, ts, args.window_us / 2))
        ri = render_bgr(sae(re, 1280, 720, ts, args.window_us / 2))
        cv2.putText(li, f"MASTER events={len(le)}", (20, 35), 0, 0.8, (0,255,0), 2)
        cv2.putText(ri, f"SLAVE events={len(re)}", (20, 35), 0, 0.8, (0,255,0), 2)
        writer.write(cv2.resize(np.hstack((li, ri)), (1280, 360)))
        frames += 1
    writer.release()
    print({"frames": frames, "master_events": counts[0], "slave_events": counts[1], "preview": str(out.resolve())})


if __name__ == "__main__": main()
