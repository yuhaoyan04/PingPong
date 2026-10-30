"""Render synchronized, trajectory-compensated ball closeups for marker review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def color_polarity(surface: np.ndarray) -> np.ndarray:
    image = np.full((surface.shape[1], surface.shape[2], 3), 16, np.uint8)
    off = np.clip(surface[0] * 255, 0, 255).astype(np.uint8)
    on = np.clip(surface[1] * 255, 0, 255).astype(np.uint8)
    image[..., 0] = np.maximum(image[..., 0], off)
    image[..., 1] = np.maximum(image[..., 1], on)
    image[..., 2] = np.maximum(image[..., 2], on)
    return image


def normalized_events(events: np.ndarray, timestamps: np.ndarray, centers: np.ndarray,
                      radii: np.ndarray, end_us: int, window_us: int,
                      size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(events):
        return (np.empty(0, np.int64),) * 3
    time = events["timestamp"].astype(np.int64)
    valid = (time <= end_us) & (time >= end_us - window_us)
    time = time[valid]
    if not len(time):
        return (np.empty(0, np.int64),) * 3
    cx = np.interp(time, timestamps, centers[:, 0]); cy = np.interp(time, timestamps, centers[:, 1])
    radius = np.interp(time, timestamps, radii)
    nx = (events["x"][valid].astype(np.float64) - cx) / radius
    ny = (events["y"][valid].astype(np.float64) - cy) / radius
    inside = nx * nx + ny * ny <= 1.0
    half = .5 * size
    u = np.rint(nx[inside] * half + half).astype(np.int64)
    v = np.rint(ny[inside] * half + half).astype(np.int64)
    return u, v, time[inside]


def time_color(events: np.ndarray, timestamps: np.ndarray, centers: np.ndarray,
               radii: np.ndarray, end_us: int, window_us: int, size: int) -> np.ndarray:
    u, v, time = normalized_events(events, timestamps, centers, radii, end_us, window_us, size)
    image = np.full((size, size, 3), 12, np.uint8)
    if len(time):
        valid = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        u, v, time = u[valid], v[valid], time[valid]
        phase = np.clip((time - (end_us - window_us)) / max(window_us, 1), 0, 1)
        # Old events are blue, recent events are red/yellow.
        image[v, u, 0] = np.clip((1.0 - phase) * 255, 0, 255).astype(np.uint8)
        image[v, u, 1] = np.clip((1.0 - np.abs(phase - .5) * 2) * 220, 0, 255).astype(np.uint8)
        image[v, u, 2] = np.clip(phase * 255, 0, 255).astype(np.uint8)
        image = cv2.dilate(image, np.ones((2, 2), np.uint8))
    return image


def load_path(path: str):
    data = np.load(path)
    return (data["timestamps_us"].astype(np.int64), data["centers_px"].astype(np.float64),
            data["radii_px"].astype(np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description="双目球心补偿局部视频")
    parser.add_argument("--master-raw", required=True); parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--master-path", required=True); parser.add_argument("--slave-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-us", type=int, default=1000)
    parser.add_argument("--time-window-us", type=int, default=2000)
    parser.add_argument("--step-us", type=int, default=1000)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--display-fps", type=float, default=25)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    paths = [load_path(args.master_path), load_path(args.slave_path)]
    start = max(path[0][0] for path in paths); end = min(path[0][-1] for path in paths)
    if end <= start:
        raise RuntimeError("The refined camera tracks have no common time interval")
    sequences = []
    longest = max(args.window_us, args.time_window_us)
    for reader in readers:
        events = reader.window(int(start - longest), int(end + 1))
        order = np.argsort(events["timestamp"], kind="stable")
        events = events[order]; sequences.append((events, events["timestamp"].astype(np.int64)))
    frame_size = (args.size * 2, args.size * 2)
    video = output / "dual_ball_compensated_detail.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, frame_size)
    contact = []
    render_times = np.arange(start, end + 1, args.step_us, dtype=np.int64)
    samples = set(np.linspace(0, len(render_times) - 1, min(10, len(render_times)), dtype=int))
    for index, timestamp in enumerate(render_times):
        rows = []
        for camera, name in enumerate(("MASTER", "SLAVE")):
            path_t, centers, radii = paths[camera]
            events, event_times = sequences[camera]
            lo = np.searchsorted(event_times, timestamp - longest, side="left")
            hi = np.searchsorted(event_times, timestamp, side="right")
            current = events[lo:hi]
            surface = trajectory_compensated_ball_patch(
                current, path_t, centers, radii, args.size, int(timestamp),
                max(100, args.window_us / 2), args.window_us)
            polarity = color_polarity(surface)
            temporal = time_color(current, path_t, centers, radii, int(timestamp),
                                  args.time_window_us, args.size)
            for tile, label in ((polarity, f"{name} polarity {args.window_us/1000:g} ms"),
                                (temporal, f"{name} time {args.time_window_us/1000:g} ms")):
                cv2.circle(tile, (args.size // 2, args.size // 2), args.size // 2 - 2,
                           (100, 100, 100), 1, cv2.LINE_AA)
                cv2.circle(tile, (args.size // 2, args.size // 2), round(args.size * .35),
                           (60, 60, 60), 1, cv2.LINE_AA)
                cv2.putText(tile, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .46,
                            (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.hstack((polarity, temporal)))
        frame = np.vstack(rows); writer.write(frame)
        if index in samples:
            contact.append(cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA))
    writer.release()
    if contact:
        blank = np.zeros_like(contact[0]); sheet=[]
        for index in range(0, len(contact), 5):
            chunk=contact[index:index+5]; sheet.append(np.hstack(chunk+[blank]*(5-len(chunk))))
        cv2.imwrite(str(output / "dual_ball_compensated_contact_sheet.jpg"), np.vstack(sheet))
    report = {"purpose":"dual_trajectory_compensated_ball_detail_no_spin",
              "common_duration_ms":float((end-start)/1000), "frames":len(render_times),
              "video":str(video.resolve()),
              "polarity_colors":"blue=OFF yellow=ON",
              "temporal_colors":"blue=old red=recent",
              "warning":"Visible arcs are marker evidence, not yet physical marker identities."}
    (output / "dual_ball_compensated_detail.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
