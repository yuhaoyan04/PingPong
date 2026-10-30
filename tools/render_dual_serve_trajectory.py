"""Render real dual-camera event views with ball trajectory overlays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae


def load_path(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "timestamps": data["timestamps_us"].astype(np.int64),
        "centers": data["centers_px"].astype(np.float64),
        "radii": data["radii_px"].astype(np.float64),
        "scores": data["scores"].astype(np.float64),
    }


def render_camera(events: np.ndarray, event_times: np.ndarray, path: dict[str, np.ndarray],
                  timestamp: int, window_us: int, width: int, height: int,
                  title: str) -> np.ndarray:
    lo = np.searchsorted(event_times, timestamp - window_us, side="left")
    hi = np.searchsorted(event_times, timestamp, side="right")
    surface = sae(events[lo:hi], width, height, timestamp, max(100, window_us / 2))
    image = render_bgr(surface)
    times = path["timestamps"]
    in_range = int(times[0]) <= timestamp <= int(times[-1])
    if in_range:
        cx = float(np.interp(timestamp, times, path["centers"][:, 0]))
        cy = float(np.interp(timestamp, times, path["centers"][:, 1]))
        radius = float(np.interp(timestamp, times, path["radii"]))
        center = (round(cx), round(cy))
        cv2.circle(image, center, max(2, round(radius)), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        past = (times <= timestamp) & (times >= timestamp - 25000)
        points = np.rint(path["centers"][past]).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False,
                          (255, 255, 0), 2, cv2.LINE_AA)
        nearest = int(np.argmin(np.abs(times - timestamp)))
        state = "MEASURED" if path["scores"][nearest] > 0 else "INTERPOLATED"
        cv2.putText(image, f"BALL {state}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "BALL OUTSIDE TRACKED INTERVAL", (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(image, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "green=circle  red=center  cyan=trajectory", (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="双相机真实事件画面与球轨迹叠加视频")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--master-path", required=True)
    parser.add_argument("--slave-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-us", type=int, default=2000)
    parser.add_argument("--step-us", type=int, default=1000)
    parser.add_argument("--display-fps", type=float, default=30.0)
    parser.add_argument("--panel-width", type=int, default=800)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    paths = [load_path(args.master_path), load_path(args.slave_path)]
    start = min(int(path["timestamps"][0]) for path in paths)
    end = max(int(path["timestamps"][-1]) for path in paths)
    sequences = []
    for reader in readers:
        events = reader.window(start - args.window_us, end + 1)
        event_times = events["timestamp"].astype(np.int64, copy=False)
        if np.any(np.diff(event_times) < 0):
            events = events[np.argsort(event_times, kind="stable")]
            event_times = events["timestamp"].astype(np.int64, copy=False)
        sequences.append((events, event_times))

    panel_width = int(args.panel_width)
    panel_height = round(panel_width * readers[0].info.height / readers[0].info.width)
    video_path = output / "real_serve_dual_trajectory.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, (2 * panel_width, panel_height))
    sample_frames = []
    timestamps = np.arange(start, end + 1, args.step_us, dtype=np.int64)
    samples = set(np.linspace(0, len(timestamps) - 1, min(12, len(timestamps)), dtype=int))
    for index, timestamp in enumerate(timestamps):
        panels = []
        for camera, title in enumerate(("MASTER actual event view", "SLAVE actual event view")):
            events, event_times = sequences[camera]
            reader = readers[camera]
            panel = render_camera(events, event_times, paths[camera], int(timestamp),
                                  args.window_us, reader.info.width, reader.info.height, title)
            panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            panels.append(panel)
        frame = np.hstack(panels)
        cv2.putText(frame, f"slow motion  source time={(timestamp-start)/1000:.1f} ms",
                    (frame.shape[1] // 2 - 180, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if index in samples:
            sample_frames.append(cv2.resize(frame, None, fx=.5, fy=.5,
                                             interpolation=cv2.INTER_AREA))
    writer.release()
    if sample_frames:
        blank = np.zeros_like(sample_frames[0])
        rows = []
        for index in range(0, len(sample_frames), 2):
            chunk = sample_frames[index:index + 2]
            rows.append(np.hstack(chunk + [blank] * (2 - len(chunk))))
        cv2.imwrite(str(output / "real_serve_dual_trajectory_contact_sheet.jpg"),
                    np.vstack(rows))
    report = {
        "purpose": "actual_dual_event_view_with_ball_trajectory_overlay",
        "source_duration_ms": float((end - start) / 1000),
        "frames": len(timestamps),
        "event_accumulation_window_us": args.window_us,
        "source_step_us": args.step_us,
        "playback_slowdown": float(1e6 / args.step_us / args.display_fps),
        "video": str(video_path.resolve()),
        "contact_sheet": str((output / "real_serve_dual_trajectory_contact_sheet.jpg").resolve()),
    }
    (output / "real_serve_dual_trajectory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
