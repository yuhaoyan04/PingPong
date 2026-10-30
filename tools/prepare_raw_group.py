"""Bridge generic monocular RAW ball detection to refinement/enhancement inputs.

Unlike batch_marker_crops.py, this has no nine-serve video timecode constants.
Only actual positive-score detections seed refinement; interpolated samples do not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(path, raw, video, camera, group, output, video_first_us, video_step_us=5000):
    path, raw, video, output = map(Path, (path, raw, video, output))
    if camera not in {"F3E", "F59"} or Path(group).name != group or not group.endswith("_" + camera):
        raise ValueError("group must be a simple name ending with matching camera")
    if not all(p.is_file() for p in (path, raw, video)) or video_step_us <= 0:
        raise ValueError("missing source inputs or invalid video time interval")
    with np.load(path) as data:
        times = data["detection_timestamps_us"].astype(np.int64)
        circles = data["measured_circles_px"].astype(float)
        scores = data["measured_scores"].astype(float)
    if (circles.shape != (len(times), 3) or scores.shape != (len(times),)
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid measured circle arrays/timebase")
    supported = np.isfinite(circles).all(axis=1) & np.isfinite(scores) & (scores > 0) & (circles[:, 2] > 0)
    coarse = np.column_stack((times[supported], circles[supported]))
    if len(coarse) < 4:
        raise ValueError("need at least four actual ball detections; predictions are not seeds")
    group_dir, coarse_dir = output / "groups" / group, output / "coarse" / group
    if group_dir.exists() or coarse_dir.exists():
        raise FileExistsError("refusing to overwrite a prepared group")
    group_dir.mkdir(parents=True)
    coarse_dir.mkdir(parents=True)
    source = {"raw": str(raw.resolve()), "video": str(video.resolve()), "camera": camera,
              "group": group, "video_first_us": int(video_first_us), "video_step_us": int(video_step_us),
              "frames": int(len(coarse)), "ball_path": str(path.resolve()),
              "warning": "Automatic ball candidates; timing, geometry and synchronization require review."}
    (group_dir / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    np.savez(coarse_dir / "candidate_tracks.npz", new=coarse)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="monocular_path.npz")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="dual-camera reference video")
    parser.add_argument("--camera", choices=["F3E", "F59"], required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-first-us", type=int, required=True, help="physical event time of first reference frame")
    parser.add_argument("--video-step-us", type=int, default=5000, help="physical sample interval, not video playback FPS")
    args = parser.parse_args()
    source = prepare(args.path, args.raw, args.video, args.camera, args.group, args.output,
                     args.video_first_us, args.video_step_us)
    print(json.dumps(source, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
