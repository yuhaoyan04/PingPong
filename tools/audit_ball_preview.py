"""Create uniform and high-activity contact sheets from a stereo preview."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sheet(frames: list[tuple[int, np.ndarray]], fps: float, columns: int = 4) -> np.ndarray:
    thumbs = []
    for index, frame in frames:
        thumb = (frame.copy() if frame.shape[:2] == (180, 640) else
                 cv2.resize(frame, (640, 180), interpolation=cv2.INTER_AREA))
        cv2.putText(thumb, f"t={index / fps:.2f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % columns:
        thumbs.append(blank.copy())
    return np.vstack([np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--output-dir")
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()
    path = Path(args.video)
    out = Path(args.output_dir) if args.output_dir else path.parent
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    scores: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Keeping hundreds of full 2560x720 frames can consume >1 GB during a
        # 1 kHz audit.  The contact sheet only needs thumbnails.
        frames.append(cv2.resize(frame, (640, 180), interpolation=cv2.INTER_AREA))
        delta = np.max(np.abs(frame.astype(np.int16) - 96), axis=2)
        scores.append(float(np.count_nonzero(delta > 25)))
    cap.release()
    if not frames:
        raise SystemExit("video has no frames")
    uniform_indices = np.linspace(0, len(frames) - 1, args.samples, dtype=int)
    gap = max(1, round(fps * 0.5))
    chosen: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if all(abs(int(index) - old) >= gap for old in chosen):
            chosen.append(int(index))
        if len(chosen) >= args.samples:
            break
    chosen.sort()
    uniform_path = out / "uniform_activity_sheet.jpg"
    peaks_path = out / "peak_activity_sheet.jpg"
    cv2.imwrite(str(uniform_path), sheet([(i, frames[i]) for i in uniform_indices], fps))
    cv2.imwrite(str(peaks_path), sheet([(i, frames[i]) for i in chosen], fps))
    print({"frames": len(frames), "fps": fps, "declared_frames": count,
           "uniform": str(uniform_path.resolve()), "peaks": str(peaks_path.resolve()),
           "peak_times_s": [round(i / fps, 3) for i in chosen]})


if __name__ == "__main__":
    main()
