"""Build a human-review video for trajectory, compensation and marker evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import (
    RawWindowReader,
    trajectory_compensated_ball_patch,
)
from propose_physical_marker_centers import polarity_midpoints


def colored(surface: np.ndarray) -> np.ndarray:
    image = np.full((surface.shape[1], surface.shape[2], 3), 18, np.uint8)
    off = np.clip(surface[0] * 255, 0, 255).astype(np.uint8)
    on = np.clip(surface[1] * 255, 0, 255).astype(np.uint8)
    image[..., 0] = np.maximum(image[..., 0], off)
    image[..., 1] = np.maximum(image[..., 1], on)
    image[..., 2] = np.maximum(image[..., 2], on)
    return image


def internal_candidates(surface: np.ndarray, inner_fraction: float,
                        maximum: int = 6,
                        percentile: float = 98.5) -> tuple[list[dict], np.ndarray]:
    """Find anonymous compact activity peaks after suppressing broad structure."""
    activity = np.maximum(surface[0], surface[1]).astype(np.float32)
    fine = cv2.GaussianBlur(activity, (0, 0), 1.8)
    broad = cv2.GaussianBlur(activity, (0, 0), 11.0)
    residual = np.maximum(fine - 1.15 * broad, 0.0)
    height, width = residual.shape
    yy, xx = np.mgrid[:height, :width]
    radius = 0.5 * min(height, width) * float(inner_fraction)
    mask = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= radius ** 2
    values = residual[mask]
    threshold = max(float(np.percentile(values, percentile)), 0.02) if values.size else 1.0
    maxima = residual == cv2.dilate(residual, np.ones((17, 17), np.uint8))
    y, x = np.where(mask & maxima & (residual >= threshold))
    if not len(x):
        return [], residual
    order = np.argsort(residual[y, x])[::-1]
    accepted: list[dict] = []
    for index in order:
        point = np.array([x[index], y[index]], np.float64)
        if any(np.linalg.norm(point - np.array([item["x"], item["y"]])) < 18
               for item in accepted):
            continue
        accepted.append({"x": float(point[0]), "y": float(point[1]),
                         "score": float(residual[y[index], x[index]])})
        if len(accepted) >= maximum:
            break
    return accepted, residual


def label(tile: np.ndarray, title: str, subtitle: str = "") -> None:
    cv2.putText(tile, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(tile, subtitle, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, .38,
                    (200, 200, 200), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="人工验收：球轨迹、逐事件运动补偿、内部标志候选")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-us", type=int, default=1000)
    parser.add_argument("--short-window-us", type=int, default=500)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--inner-fraction", type=float, default=.72)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--candidate-percentile", type=float, default=98.5)
    parser.add_argument("--marker-inner-fraction", type=float, default=.36,
                        help="marker search radius relative to the square patch; .36 is 72%% of the ball radius")
    parser.add_argument("--marker-nms-px", type=int, default=35)
    parser.add_argument("--marker-support-px", type=float, default=40.0)
    parser.add_argument("--display-fps", type=float, default=20.0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = np.load(args.path)
    timestamps = path["timestamps_us"].astype(np.int64)
    centers = path["centers_px"].astype(np.float64)
    radii = path["radii_px"].astype(np.float64)
    reader = RawWindowReader(args.raw)
    longest = max(args.window_us, args.short_window_us)
    events = reader.window(int(timestamps[0] - longest), int(timestamps[-1] + 1))
    event_times = events["timestamp"].astype(np.int64, copy=False)
    if np.any(np.diff(event_times) < 0):
        events = events[np.argsort(event_times, kind="stable")]
        event_times = events["timestamp"].astype(np.int64, copy=False)

    size = int(args.patch_size)
    video_path = output / "trajectory_compensation_marker_review.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, (size * 3, size))
    rows = []
    samples = []
    sample_indices = set(np.linspace(0, len(timestamps) - 1,
                                     min(16, len(timestamps)), dtype=int))
    for index, timestamp in enumerate(timestamps):
        surfaces = []
        counts = []
        for window_us in (args.short_window_us, args.window_us):
            lo = np.searchsorted(event_times, timestamp - window_us, side="left")
            hi = np.searchsorted(event_times, timestamp, side="right")
            current = events[lo:hi]
            surface = trajectory_compensated_ball_patch(
                current, timestamps, centers, radii, size, int(timestamp),
                max(100.0, window_us / 2), window_us)
            surfaces.append(surface)
            counts.append(len(current))

        short = colored(surfaces[0])
        long = colored(surfaces[1])
        region_candidates, residual = internal_candidates(
            surfaces[1], args.inner_fraction, args.max_candidates,
            args.candidate_percentile)
        # A physical dark dot generates a neighbouring OFF/ON pair as it
        # rotates.  Their midpoint is a much better centre hypothesis than an
        # arbitrary maximum of the event-density image.  Keep the search well
        # inside the projected sphere to suppress the outer silhouette.
        paired_candidates = polarity_midpoints(
            long, args.marker_nms_px, args.marker_support_px,
            args.marker_inner_fraction, args.max_candidates)
        paired_candidates = [item for item in paired_candidates
                             if item["assigned_active_pixels"] >= 2
                             and 1.5 <= item["polarity_separation_px"]
                             <= args.marker_support_px]
        enhanced = np.clip(residual / max(float(np.percentile(
            residual, 99.5)), 1e-6) * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(mask, (size // 2, size // 2),
                   round(size * .5 * args.inner_fraction), 255, -1)
        enhanced[mask == 0] = 10
        for item in region_candidates:
            point = (round(item["x"]), round(item["y"]))
            cv2.circle(long, point, 6, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(enhanced, point, 5, (255, 255, 0), 1, cv2.LINE_AA)
        for number, item in enumerate(paired_candidates, 1):
            off = (round(item["off_x"]), round(item["off_y"]))
            on = (round(item["on_x"]), round(item["on_y"]))
            point = (round(item["x"]), round(item["y"]))
            for tile in (long, enhanced):
                cv2.line(tile, off, on, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.circle(tile, off, 3, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(tile, on, 3, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(tile, point, (0, 255, 0), cv2.MARKER_CROSS,
                               11, 1, cv2.LINE_AA)
            cv2.putText(enhanced, str(number), (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 255, 0), 1, cv2.LINE_AA)
        for tile in (short, long):
            cv2.circle(tile, (size // 2, size // 2), size // 2 - 1, (90, 90, 90), 1)
            cv2.circle(tile, (size // 2, size // 2),
                       round(size * .5 * args.inner_fraction), (70, 70, 70), 1)
        label(short, f"A compensated {args.short_window_us/1000:g} ms",
              "blue=OFF yellow=ON")
        label(long, f"B compensated {args.window_us/1000:g} ms",
              f"red=compact regions {len(region_candidates)}")
        label(enhanced, "C point evidence",
              f"cyan=region green=paired {len(paired_candidates)}")
        frame = np.hstack((short, long, enhanced))
        cv2.putText(frame, f"index={index}  t={(timestamp-timestamps[0])/1000:.1f} ms",
                    (7, size - 8), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if index in sample_indices:
            samples.append(frame.copy())
        rows.append({
            "index": index, "timestamp_us": int(timestamp),
            "center_x": float(centers[index, 0]),
            "center_y": float(centers[index, 1]),
            "radius_px": float(radii[index]),
            "short_window_events": int(counts[0]),
            "long_window_events": int(counts[1]),
            "compact_region_candidates": len(region_candidates),
            "paired_marker_candidates": len(paired_candidates),
        })
    writer.release()
    if samples:
        thumb_size = (round(size * 3 * .5), round(size * .5))
        thumbs = [cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)
                  for frame in samples]
        blank = np.zeros_like(thumbs[0])
        sheet_rows = []
        for start in range(0, len(thumbs), 2):
            chunk = thumbs[start:start + 2]
            sheet_rows.append(np.hstack(chunk + [blank] * (2 - len(chunk))))
        cv2.imwrite(str(output / "trajectory_compensation_marker_contact_sheet.jpg"),
                    np.vstack(sheet_rows))
    with (output / "trajectory_compensation_marker_review.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader(); writer_csv.writerows(rows)
    paired_counts = np.asarray(
        [row["paired_marker_candidates"] for row in rows], np.float64)
    region_counts = np.asarray(
        [row["compact_region_candidates"] for row in rows], np.float64)
    report = {
        "purpose": "human_review_of_trajectory_exact_compensation_and_marker_evidence",
        "frames": len(rows),
        "duration_ms": float((timestamps[-1] - timestamps[0]) / 1000),
        "normalization": "per-event interpolated centre and radius",
        "inner_radius_fraction": args.inner_fraction,
        "marker_search_radius_fraction_of_patch": args.marker_inner_fraction,
        "marker_search_radius_fraction_of_ball": 2.0 * args.marker_inner_fraction,
        "marker_definition": "midpoint of a local paired OFF/ON event response",
        "candidate_percentile": args.candidate_percentile,
        "compact_region_count_median": float(np.median(region_counts)),
        "paired_marker_count_median": float(np.median(paired_counts)),
        "paired_marker_count_p10": float(np.percentile(paired_counts, 10)),
        "paired_marker_count_p90": float(np.percentile(paired_counts, 90)),
        "paired_marker_frame_fraction": float(np.mean(paired_counts > 0)),
        "video": str(video_path.resolve()),
        "contact_sheet": str((output / "trajectory_compensation_marker_contact_sheet.jpg").resolve()),
        "csv": str((output / "trajectory_compensation_marker_review.csv").resolve()),
        "acceptance_questions": [
            "Does the normalized ball remain centred with a stable outer radius?",
            "Are compact internal structures visible for multiple consecutive frames?",
            "Do green candidates follow those structures rather than the outer crescent?",
        ],
        "warning": "Compact regions and paired centres are anonymous hypotheses, not cross-frame IDs or spin measurements.",
    }
    (output / "trajectory_compensation_marker_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
