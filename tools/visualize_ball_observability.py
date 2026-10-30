"""Visualize ball ROI and anonymous texture candidates without estimating spin."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, motion_compensated_ball_patch


def feature_candidates(gray: np.ndarray, maximum: int, radius_fraction: float):
    """Return anonymous peaks of local event density, not physical dot IDs."""
    mask = np.zeros_like(gray)
    radius = round(gray.shape[0] * radius_fraction)
    cv2.circle(mask, (gray.shape[1] // 2, gray.shape[0] // 2), radius, 255, -1)
    density = cv2.GaussianBlur((gray > 10).astype(np.float32), (0, 0), 3)
    values = density[mask > 0]
    threshold = max(float(np.percentile(values, 92)), .025)
    local_maximum = density == cv2.dilate(density, np.ones((15, 15), np.uint8))
    y, x = np.where(local_maximum & (density >= threshold) & (mask > 0))
    if not len(x):
        return np.empty((0, 2), np.float32)
    order = np.argsort(density[y, x])[::-1][:maximum]
    return np.column_stack((x[order], y[order])).astype(np.float32)


def colored_patch(patch: np.ndarray) -> np.ndarray:
    image = np.full((patch.shape[1], patch.shape[2], 3), 24, np.uint8)
    off = np.clip(patch[0] * 255, 0, 255).astype(np.uint8)
    on = np.clip(patch[1] * 255, 0, 255).astype(np.uint8)
    image[..., 0] = np.maximum(image[..., 0], off)
    image[..., 1] = np.maximum(image[..., 1], on)
    image[..., 2] = np.maximum(image[..., 2], on)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="逐窗查看球ROI和匿名纹理候选；不计算RPM")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--paths", required=True, help="Contains master_path.npz and slave_path.npz")
    parser.add_argument("--first-index", type=int, required=True)
    parser.add_argument("--last-index", type=int, required=True)
    parser.add_argument("--windows-us", type=int, nargs="+", default=[250, 500, 1000, 2500, 5000])
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--display-fps", type=float, default=25)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path_dir = Path(args.paths)
    paths = [np.load(path_dir / "master_path.npz"), np.load(path_dir / "slave_path.npz")]
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    windows = sorted(set(args.windows_us))
    size = args.patch_size
    camera_names = ["MASTER", "SLAVE"]
    sequences = []
    for reader, path in zip(readers, paths):
        timestamps = path["timestamps_us"].astype(np.int64)
        centers = path["centers_px"].astype(np.float64)
        radii = path["radii_px"].astype(np.float64)
        dt = np.gradient(timestamps).astype(np.float64) * 1e-6
        velocity = np.column_stack([
            np.gradient(centers[:, axis]) / dt for axis in range(2)
        ])
        events = reader.window(
            int(timestamps[args.first_index] - max(windows)),
            int(timestamps[args.last_index] + 1),
        )
        event_times = events["timestamp"].astype(np.int64, copy=False)
        if np.any(np.diff(event_times) < 0):
            events = events[np.argsort(event_times, kind="stable")]
            event_times = events["timestamp"].astype(np.int64, copy=False)
        sequences.append((timestamps, centers, radii, velocity, events, event_times))

    frame_size = (len(windows) * size, 2 * size)
    video_path = output / "ball_observability.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.display_fps, frame_size
    )
    fields = [
        "path_index", "timestamp_us", "camera", "window_us", "roi_event_count",
        "active_pixel_count", "texture_candidate_count", "center_x", "center_y", "radius_px",
    ]
    rows = []
    sample_indices = set(np.linspace(
        args.first_index, args.last_index, min(12, args.last_index - args.first_index + 1),
        dtype=int,
    ).tolist())
    sample_frames = []
    for index in range(args.first_index, args.last_index + 1):
        camera_rows = []
        timestamp = int(paths[0]["timestamps_us"][index])
        for camera, sequence in enumerate(sequences):
            timestamps, centers, radii, velocity, events, event_times = sequence
            tiles = []
            for window_us in windows:
                lo = np.searchsorted(event_times, timestamp - window_us, side="left")
                hi = np.searchsorted(event_times, timestamp, side="right")
                current = events[lo:hi]
                age_s = (timestamp - current["timestamp"].astype(np.int64)) * 1e-6
                compensated_x = current["x"].astype(np.float64) + velocity[index, 0] * age_s
                compensated_y = current["y"].astype(np.float64) + velocity[index, 1] * age_s
                roi_inside = (
                    (compensated_x - centers[index, 0]) ** 2
                    + (compensated_y - centers[index, 1]) ** 2
                    <= radii[index] ** 2
                )
                roi_event_count = int(np.count_nonzero(roi_inside))
                patch = motion_compensated_ball_patch(
                    current[roi_inside], centers[index], radii[index], size, timestamp,
                    max(100, window_us / 2), velocity[index], window_us,
                )
                gray = np.clip(np.maximum(patch[0], patch[1]) * 255, 0, 255).astype(np.uint8)
                points = feature_candidates(gray, args.max_candidates, .38)
                tile = colored_patch(patch)
                cv2.circle(tile, (size // 2, size // 2), round(size * .5), (80, 80, 80), 1)
                for point in points:
                    cv2.circle(tile, tuple(np.rint(point).astype(int)), 3, (0, 0, 255), 1)
                cv2.putText(tile, f"{camera_names[camera]} {window_us/1000:g}ms", (5, 17),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(tile, f"events={roi_event_count} cand={len(points)}", (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1, cv2.LINE_AA)
                active = int(np.count_nonzero(np.max(patch, axis=0)))
                rows.append({
                    "path_index": index, "timestamp_us": timestamp,
                    "camera": camera_names[camera].lower(), "window_us": window_us,
                    "roi_event_count": roi_event_count, "active_pixel_count": active,
                    "texture_candidate_count": len(points),
                    "center_x": centers[index, 0], "center_y": centers[index, 1],
                    "radius_px": radii[index],
                })
                tiles.append(tile)
            camera_rows.append(np.hstack(tiles))
        frame = np.vstack(camera_rows)
        cv2.putText(frame, f"index={index}  t={timestamp} us  red=anonymous texture candidate",
                    (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if index in sample_indices:
            small = cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            sample_frames.append(small)
            cv2.imwrite(str(output / f"sample_{index:03d}.png"), frame)
    writer.release()

    with (output / "observability.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    if sample_frames:
        cv2.imwrite(str(output / "observability_contact_sheet.jpg"), np.vstack(sample_frames))

    aggregates = []
    for camera in camera_names:
        for window_us in windows:
            group = [row for row in rows if row["camera"] == camera.lower()
                     and row["window_us"] == window_us]
            candidate_counts = np.asarray(
                [row["texture_candidate_count"] for row in group], np.float64)
            aggregates.append({
                "camera": camera.lower(), "window_us": window_us,
                "frames": len(group),
                "roi_event_count_median": float(np.median(
                    [row["roi_event_count"] for row in group])),
                "active_pixel_count_median": float(np.median(
                    [row["active_pixel_count"] for row in group])),
                "texture_candidate_count_median": float(np.median(candidate_counts)),
                "texture_candidate_count_p10": float(np.percentile(candidate_counts, 10)),
                "texture_candidate_count_p90": float(np.percentile(candidate_counts, 90)),
                "fraction_at_candidate_limit": float(np.mean(
                    candidate_counts == args.max_candidates)),
            })
    summary = {
        "purpose": "ball_and_texture_observability_only_no_spin_estimation",
        "first_index": args.first_index,
        "last_index": args.last_index,
        "windows_us": windows,
        "video": str(video_path.resolve()),
        "contact_sheet": str((output / "observability_contact_sheet.jpg").resolve()),
        "csv": str((output / "observability.csv").resolve()),
        "candidate_method": "local maxima of Gaussian-smoothed active-event density",
        "aggregates": aggregates,
        "warning": "Red points are anonymous event-density peaks, not identified physical dots or dot centres.",
    }
    (output / "observability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
