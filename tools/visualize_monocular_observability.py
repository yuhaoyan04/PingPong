"""Normalize one tracked ball and prepare marker-observability research assets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, motion_compensated_ball_patch
from visualize_ball_observability import colored_patch, feature_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单目球内纹理逐窗可视化与人工标注包；不计算 RPM")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--windows-us", type=int, nargs="+",
                        default=[250, 500, 1000, 2500, 5000])
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--last-index", type=int,
                        help="inclusive; defaults to the final path index")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--display-fps", type=float, default=25)
    parser.add_argument("--annotation-frames", type=int, default=30)
    parser.add_argument("--annotation-window-us", type=int, default=1000)
    parser.add_argument("--annotation-start-index", type=int,
                        help="local path index for a contiguous/strided annotation sequence")
    parser.add_argument("--annotation-stride", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotation_dir = output / "annotation_pack"
    annotation_dir.mkdir(exist_ok=True)
    loaded = np.load(args.path)
    all_timestamps = loaded["timestamps_us"].astype(np.int64)
    all_centers = loaded["centers_px"].astype(np.float64)
    all_radii = loaded["radii_px"].astype(np.float64)
    dt = np.gradient(all_timestamps).astype(np.float64) * 1e-6
    all_velocity = np.column_stack([
        np.gradient(all_centers[:, axis]) / dt for axis in range(2)
    ])
    first = max(0, args.first_index)
    last = min(len(all_timestamps) - 1,
               args.last_index if args.last_index is not None else len(all_timestamps) - 1)
    if last < first:
        raise ValueError("last-index must be greater than or equal to first-index")
    selected = slice(first, last + 1)
    timestamps = all_timestamps[selected]
    centers = all_centers[selected]
    radii = all_radii[selected]
    velocity = all_velocity[selected]
    reader = RawWindowReader(args.raw)
    windows = sorted(set(args.windows_us))
    events = reader.window(int(timestamps[0] - max(windows)), int(timestamps[-1] + 1))
    event_times = events["timestamp"].astype(np.int64, copy=False)
    if np.any(np.diff(event_times) < 0):
        events = events[np.argsort(event_times, kind="stable")]
        event_times = events["timestamp"].astype(np.int64, copy=False)

    size = args.patch_size
    video_path = output / "monocular_observability.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, (len(windows) * size, size))
    sample_indices = set(np.linspace(0, len(timestamps) - 1,
                                     min(12, len(timestamps)), dtype=int))
    if args.annotation_start_index is None:
        annotation_indices = set(np.linspace(
            max(0, len(timestamps) // 20),
            max(0, len(timestamps) - 1 - len(timestamps) // 20),
            min(args.annotation_frames, len(timestamps)), dtype=int))
        annotation_selection = "spread"
    else:
        annotation_indices = set(range(
            max(0, args.annotation_start_index), len(timestamps),
            max(1, args.annotation_stride)))
        annotation_indices = set(sorted(annotation_indices)[:args.annotation_frames])
        annotation_selection = "strided"
    contact_frames = []
    annotation_images = []
    annotation_rows = []
    rows = []
    for index, timestamp in enumerate(timestamps):
        tiles = []
        annotation_image = None
        for window_us in windows:
            lo = np.searchsorted(event_times, timestamp - window_us, side="left")
            hi = np.searchsorted(event_times, timestamp, side="right")
            current = events[lo:hi]
            age_s = (timestamp - current["timestamp"].astype(np.int64)) * 1e-6
            compensated_x = current["x"].astype(np.float64) + velocity[index, 0] * age_s
            compensated_y = current["y"].astype(np.float64) + velocity[index, 1] * age_s
            inside = ((compensated_x - centers[index, 0]) ** 2
                      + (compensated_y - centers[index, 1]) ** 2 <= radii[index] ** 2)
            patch = motion_compensated_ball_patch(
                current[inside], centers[index], radii[index], size, int(timestamp),
                max(100, window_us / 2), velocity[index], window_us)
            gray = np.clip(np.max(patch, axis=0) * 255, 0, 255).astype(np.uint8)
            points = feature_candidates(gray, args.max_candidates, .38)
            raw_image = colored_patch(patch)
            if window_us == args.annotation_window_us:
                annotation_image = raw_image.copy()
            tile = raw_image.copy()
            cv2.circle(tile, (size // 2, size // 2), size // 2 - 1, (80, 80, 80), 1)
            for point in points:
                cv2.circle(tile, tuple(np.rint(point).astype(int)), 3, (0, 0, 255), 1)
            cv2.putText(tile, f"{window_us/1000:g} ms", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, f"events={np.count_nonzero(inside)} cand={len(points)}", (5, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append({
                "path_index": first + index,
                "local_path_index": index,
                "timestamp_us": int(timestamp),
                "window_us": window_us,
                "roi_event_count": int(np.count_nonzero(inside)),
                "active_pixel_count": int(np.count_nonzero(gray)),
                "anonymous_candidate_count": len(points),
                "center_x": centers[index, 0],
                "center_y": centers[index, 1],
                "radius_px": radii[index],
            })
            tiles.append(tile)
        frame = np.hstack(tiles)
        cv2.putText(frame, f"index={index} red=anonymous density peak; no marker ID",
                    (8, size - 8), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1,
                    cv2.LINE_AA)
        writer.write(frame)
        if index in sample_indices:
            contact_frames.append(cv2.resize(frame, None, fx=.5, fy=.5,
                                             interpolation=cv2.INTER_AREA))
        if index in annotation_indices and annotation_image is not None:
            name = f"frame_{index:04d}.png"
            cv2.imwrite(str(annotation_dir / name), annotation_image)
            labeled_annotation = annotation_image.copy()
            cv2.putText(labeled_annotation, f"local={index} source={first + index}",
                        (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (255, 255, 255), 1, cv2.LINE_AA)
            annotation_images.append(labeled_annotation)
            annotation_rows.append({
                "frame": name,
                "path_index": int(first + index),
                "local_path_index": int(index),
                "timestamp_us": int(timestamp),
                "window_us": args.annotation_window_us,
                "center_x_full_image": float(centers[index, 0]),
                "center_y_full_image": float(centers[index, 1]),
                "radius_px_full_image": float(radii[index]),
                "labels": [],
                "status": "unreviewed",
            })
    writer.release()
    if contact_frames:
        cv2.imwrite(str(output / "monocular_observability_contact_sheet.jpg"),
                    np.vstack(contact_frames))
    if annotation_images:
        columns = 5
        blank = np.zeros_like(annotation_images[0])
        annotation_sheet_rows = [
            np.hstack(annotation_images[i:i + columns]
                      + [blank] * (columns - len(annotation_images[i:i + columns])))
            for i in range(0, len(annotation_images), columns)
        ]
        cv2.imwrite(str(annotation_dir / "annotation_contact_sheet.jpg"),
                    np.vstack(annotation_sheet_rows))

    fields = list(rows[0])
    with (output / "monocular_observability.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    (annotation_dir / "annotations.json").write_text(
        json.dumps({
            "coordinate_system": "normalized_patch_pixels",
            "image_size_px": [size, size],
            "label_definition": "centre of a visually identifiable physical ball marker",
            "left_click": "add marker centre",
            "right_click": "remove nearest marker centre",
            "warning": "Event arcs are observations; do not label anonymous edge fragments as separate dots.",
            "frames": annotation_rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    aggregates = []
    for window_us in windows:
        group = [row for row in rows if row["window_us"] == window_us]
        counts = np.asarray([row["anonymous_candidate_count"] for row in group])
        aggregates.append({
            "window_us": window_us,
            "frames": len(group),
            "roi_event_count_median": float(np.median([row["roi_event_count"] for row in group])),
            "active_pixel_count_median": float(np.median([row["active_pixel_count"] for row in group])),
            "anonymous_candidate_count_median": float(np.median(counts)),
            "anonymous_candidate_count_p10": float(np.percentile(counts, 10)),
            "anonymous_candidate_count_p90": float(np.percentile(counts, 90)),
        })
    report = {
        "purpose": "monocular_marker_observability_and_annotation_pack_no_spin",
        "frames": len(timestamps),
        "source_first_index": first,
        "source_last_index": last,
        "windows_us": windows,
        "annotation_frames": len(annotation_rows),
        "annotation_window_us": args.annotation_window_us,
        "annotation_selection": annotation_selection,
        "annotation_start_index": args.annotation_start_index,
        "annotation_stride": args.annotation_stride,
        "aggregates": aggregates,
        "video": str(video_path.resolve()),
        "contact_sheet": str((output / "monocular_observability_contact_sheet.jpg").resolve()),
        "annotations": str((annotation_dir / "annotations.json").resolve()),
        "warning": "Automatic red candidates are anonymous density peaks, not physical marker centres or identities.",
    }
    (output / "monocular_observability_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
