"""Propose physical dot centres by pairing OFF/ON event lobes.

The coloured annotation patches contain blue OFF events and yellow ON events.
A moving dark dot normally produces two partial arcs rather than a filled disc.
This tool first finds sparse activity regions, assigns nearby events to each
region, and places the proposed physical centre midway between the two polarity
centroids.  Proposals remain anonymous and require human identity review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from propose_marker_regions import propose


def polarity_midpoints(image: np.ndarray, nms_px: int, support_px: float,
                       inner_radius_fraction: float, maximum: int) -> list[dict]:
    height, width = image.shape[:2]
    seeds = propose(image, 5.0, 90.0, nms_px, inner_radius_fraction, maximum)
    if not seeds:
        return []

    yy, xx = np.mgrid[:height, :width]
    radius = min(height, width) * inner_radius_fraction
    inside = np.hypot(xx - width / 2, yy - height / 2) < radius
    active = (np.max(image, axis=2) > 45) & inside
    coordinates = np.column_stack((xx[active], yy[active])).astype(np.float64)
    seed_xy = np.asarray([[item["x"], item["y"]] for item in seeds], np.float64)
    distances = np.linalg.norm(coordinates[:, None, :] - seed_xy[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    nearest = np.min(distances, axis=1)
    pixels = image[active].astype(np.float64)
    results = []
    for index, seed in enumerate(seeds):
        selected = (assignments == index) & (nearest <= support_px)
        if not np.any(selected):
            continue
        xy = coordinates[selected]
        values = pixels[selected]
        off_weights = np.maximum(values[:, 0] - 24.0, 0.0)
        on_weights = np.maximum(values[:, 1] - 24.0, 0.0)
        off_weight = float(np.sum(off_weights))
        on_weight = float(np.sum(on_weights))
        if off_weight <= 0.0 or on_weight <= 0.0:
            continue
        off_center = np.sum(xy * off_weights[:, None], axis=0) / off_weight
        on_center = np.sum(xy * on_weights[:, None], axis=0) / on_weight
        center = (off_center + on_center) / 2.0
        separation = float(np.linalg.norm(off_center - on_center))
        if separation > support_px:
            continue
        results.append({
            "x": float(center[0]),
            "y": float(center[1]),
            "off_x": float(off_center[0]),
            "off_y": float(off_center[1]),
            "on_x": float(on_center[0]),
            "on_y": float(on_center[1]),
            "polarity_separation_px": separation,
            "assigned_active_pixels": int(np.count_nonzero(selected)),
            "source": "paired_polarity_midpoint",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair blue/yellow event lobes and propose anonymous physical-dot centres")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nms-px", type=int, default=45)
    parser.add_argument("--support-px", type=float, default=48.0)
    parser.add_argument("--inner-radius-fraction", type=float, default=.40)
    parser.add_argument("--maximum", type=int, default=12)
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    output = args.output or args.annotations.parent / "physical_marker_center_proposals.json"
    overlay_dir = output.parent / "physical_center_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records = []
    overlays = []
    for frame in payload["frames"]:
        image = cv2.imread(str(args.annotations.parent / frame["frame"]))
        if image is None:
            raise RuntimeError(f"could not read {args.annotations.parent / frame['frame']}")
        proposals = polarity_midpoints(
            image, args.nms_px, args.support_px,
            args.inner_radius_fraction, args.maximum)
        overlay = image.copy()
        for number, point in enumerate(proposals, 1):
            off = (round(point["off_x"]), round(point["off_y"]))
            on = (round(point["on_x"]), round(point["on_y"]))
            center = (round(point["x"]), round(point["y"]))
            cv2.line(overlay, off, on, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(overlay, off, 4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(overlay, on, 4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, center, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(overlay, f"C{number}", (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"physical-centre proposals={len(proposals)}",
                    (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / frame["frame"]), overlay)
        overlays.append(overlay)
        records.append({
            "frame": frame["frame"],
            "path_index": frame["path_index"],
            "timestamp_us": frame["timestamp_us"],
            "proposals": proposals,
        })

    if overlays:
        columns = 5
        blank = np.zeros_like(overlays[0])
        rows = [np.hstack(overlays[i:i + columns]
                          + [blank] * (columns - len(overlays[i:i + columns])))
                for i in range(0, len(overlays), columns)]
        cv2.imwrite(str(output.parent / "physical_marker_center_contact_sheet.jpg"),
                    np.vstack(rows))
    counts = np.asarray([len(record["proposals"]) for record in records], np.float64)
    report = {
        "purpose": "anonymous_physical_dot_centre_proposals_for_human_review",
        "method": "midpoint of locally paired OFF/ON event centroids",
        "parameters": {
            "nms_px": args.nms_px,
            "support_px": args.support_px,
            "inner_radius_fraction_of_image": args.inner_radius_fraction,
            "maximum": args.maximum,
        },
        "frames": len(records),
        "proposal_count_median": float(np.median(counts)) if len(counts) else 0.0,
        "proposal_count_p10": float(np.percentile(counts, 10)) if len(counts) else 0.0,
        "proposal_count_p90": float(np.percentile(counts, 90)) if len(counts) else 0.0,
        "records": records,
        "limitations": [
            "proposals are anonymous and do not establish cross-camera identity",
            "partial visibility and overlapping event lobes can bias a centre",
            "human review and stereo epipolar validation remain required",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
