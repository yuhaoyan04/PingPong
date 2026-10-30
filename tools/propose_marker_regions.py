"""Propose merged marker regions inside normalized event-ball patches.

The proposals assist annotation.  They are not marker identities or ground truth.
The conservative inner mask intentionally excludes the ambiguous silhouette band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def propose(image: np.ndarray, sigma: float, percentile: float, nms_px: int,
            inner_radius_fraction: float, maximum: int) -> list[dict]:
    active = (np.max(image, axis=2) > 45).astype(np.float32)
    height, width = active.shape
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = min(height, width) * inner_radius_fraction
    inside = np.hypot(x_grid - width / 2, y_grid - height / 2) < radius
    density = cv2.GaussianBlur(active, (0, 0), sigma)
    values = density[inside]
    threshold = max(float(np.percentile(values, percentile)), .01)
    kernel_size = max(3, int(nms_px) | 1)
    local = density == cv2.dilate(density, np.ones((kernel_size, kernel_size), np.uint8))
    y, x = np.where(local & inside & (density >= threshold))
    order = np.argsort(density[y, x])[::-1][:maximum]
    return [{"x": float(x[i]), "y": float(y[i]), "density": float(density[y[i], x[i]])}
            for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成保守的球面标记区域候选，不赋予点 ID")
    parser.add_argument("annotations", help="annotation_pack/annotations.json")
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--nms-px", type=int, default=25)
    parser.add_argument("--inner-radius-fraction", type=float, default=.40,
                        help="relative to square size; .40 is 80%% of the ball radius")
    parser.add_argument("--maximum", type=int, default=16)
    args = parser.parse_args()

    manifest = Path(args.annotations)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    overlays = []
    overlay_dir = manifest.parent / "proposal_overlays"
    overlay_dir.mkdir(exist_ok=True)
    for frame in payload["frames"]:
        image = cv2.imread(str(manifest.parent / frame["frame"]))
        proposals = propose(image, args.sigma, args.percentile, args.nms_px,
                            args.inner_radius_fraction, args.maximum)
        overlay = image.copy()
        for number, point in enumerate(proposals, 1):
            center = (round(point["x"]), round(point["y"]))
            cv2.circle(overlay, center, 5, (0, 0, 255), 1)
            cv2.putText(overlay, str(number), (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
        cv2.putText(overlay, f"regions={len(proposals)} (anonymous proposals)",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
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
        cv2.imwrite(str(manifest.parent / "marker_region_proposals_contact_sheet.jpg"),
                    np.vstack(rows))
    counts = np.asarray([len(record["proposals"]) for record in records])
    report = {
        "purpose": "annotation_assistance_only_anonymous_marker_region_proposals",
        "method": "Gaussian density maxima after conservative silhouette-band exclusion",
        "parameters": {
            "sigma": args.sigma,
            "percentile": args.percentile,
            "nms_px": args.nms_px,
            "inner_radius_fraction_of_image": args.inner_radius_fraction,
            "maximum": args.maximum,
        },
        "frames": len(records),
        "proposal_count_median": float(np.median(counts)),
        "proposal_count_p10": float(np.percentile(counts, 10)),
        "proposal_count_p90": float(np.percentile(counts, 90)),
        "records": records,
        "limitations": [
            "proposals have no cross-frame identity",
            "one physical marker can still yield zero or multiple proposals",
            "markers near the projected ball silhouette are intentionally excluded",
            "manual labels are required before measuring detection accuracy",
        ],
    }
    output = manifest.parent / "marker_region_proposals.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
