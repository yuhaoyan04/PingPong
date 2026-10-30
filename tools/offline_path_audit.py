"""Fit smooth full-flight image paths from ambiguous per-bin Hough circles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.config import load_config
from pingpong_event_pipeline.detector import HoughCircleBallDetector
from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae
from pingpong_event_pipeline.offline_path import fit_smooth_candidate_path


def analyse(reader, start: int, end: int, bin_us: int, detector, label: str):
    events = reader.window(start, end)
    event_times = events["timestamp"].astype(np.int64)
    timestamps = np.arange(start + bin_us, end + 1, bin_us, dtype=np.int64)
    candidate_bins = []; frames = []
    for timestamp in timestamps:
        lo = np.searchsorted(event_times, timestamp - bin_us, side="right")
        hi = np.searchsorted(event_times, timestamp, side="right")
        surface = sae(events[lo:hi], reader.info.width, reader.info.height,
                      int(timestamp), 0.5 * bin_us)
        candidate_bins.append(detector.detect(surface, int(timestamp), None))
        frames.append(render_bgr(surface))
    path = fit_smooth_candidate_path(timestamps, candidate_bins)
    for i, image in enumerate(frames):
        center = tuple(np.rint(path.centers_px[i]).astype(int))
        cv2.circle(image, center, max(2, round(path.radii_px[i])), (0, 255, 0), 2)
        if path.selected[i] >= 0:
            candidate = sorted(candidate_bins[i], key=lambda item: item.score, reverse=True)[:15][path.selected[i]]
            cv2.circle(image, (round(candidate.x), round(candidate.y)), 4, (0, 0, 255), -1)
        cv2.putText(image, f"{label} {(timestamps[i]-start)*1e-3:.1f} ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return timestamps, candidate_bins, path, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-raw", required=True)
    ap.add_argument("--slave-raw", required=True)
    ap.add_argument("--config", default="config/offline_fast_ball.yaml")
    ap.add_argument("--start-offset-s", type=float, required=True)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--bin-us", type=int, default=1000)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    left, right = RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)
    common = max(left.info.start_us, right.info.start_us)
    start = common + round(args.start_offset_s * 1e6)
    end = start + round(args.duration_s * 1e6)
    results = []
    for reader, label in ((left, "MASTER"), (right, "SLAVE")):
        results.append(analyse(reader, start, end, args.bin_us,
                               HoughCircleBallDetector(cfg["detection"]), label))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frames = [np.hstack((a, b)) for a, b in zip(results[0][3], results[1][3])]
    writer = cv2.VideoWriter(str(out / "smooth_paths.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             1e6 / args.bin_us, (left.info.width * 2, left.info.height))
    for image in frames:
        writer.write(image)
    writer.release()
    payload = {}
    for name, result in zip(("master", "slave"), results):
        timestamps, candidate_bins, path, _ = result
        np.savez(out / f"{name}_path.npz", timestamps_us=timestamps,
                 centers_px=path.centers_px, radii_px=path.radii_px,
                 selected=path.selected, residual_px=path.residual_px)
        valid = np.isfinite(path.residual_px) & (path.selected >= 0)
        payload[name] = {
            "bins": len(timestamps), "bins_with_candidates": sum(bool(x) for x in candidate_bins),
            "path_inliers": int(np.count_nonzero(valid)),
            "median_candidate_residual_px": float(np.median(path.residual_px[valid])),
            "p90_candidate_residual_px": float(np.percentile(path.residual_px[valid], 90)),
        }
    payload["video"] = str((out / "smooth_paths.mp4").resolve())
    (out / "path_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
