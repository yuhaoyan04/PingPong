"""独立验证双目标定的校正后极线误差，并输出可视化。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_stereo import corners  # noqa: E402


def blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 8; p.maxThreshold = 255
    p.filterByArea = True; p.minArea = 12; p.maxArea = 20000
    p.filterByCircularity = True; p.minCircularity = 0.12
    p.filterByConvexity = False; p.filterByInertia = False
    p.filterByColor = True; p.blobColor = 255
    return cv2.SimpleBlobDetector_create(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    ap.add_argument("--images", default="calibration/captures")
    ap.add_argument("--output", default="calibration/stereo_validation.json")
    ap.add_argument("--preview", default="calibration/rectified_check.png")
    args = ap.parse_args()

    data = np.load(args.calibration)
    K1, D1, K2, D2 = data["K1"], data["D1"], data["K2"], data["D2"]
    R1, R2, P1, P2 = data["R1"], data["R2"], data["P1"], data["P2"]
    size = tuple(int(v) for v in data["image_size"])
    report_path = Path(args.calibration).with_suffix(".json")
    calibration_report = json.loads(report_path.read_text(encoding="utf-8"))
    ids = calibration_report["used_pair_ids"]
    root, detector = Path(args.images), blob_detector()

    all_dy, pair_stats = [], []
    preview_source = None
    for pair_id in ids:
        left, lc = corners(root / f"master_{pair_id}.png", (3, 4), "led-circles", detector)
        right, rc = corners(root / f"slave_{pair_id}.png", (3, 4), "led-circles", detector)
        if lc is None or rc is None:
            continue
        if float(np.dot(lc[-1, 0] - lc[0, 0], rc[-1, 0] - rc[0, 0])) < 0:
            rc = rc[::-1].copy()
        ul = cv2.undistortPoints(lc, K1, D1, R=R1, P=P1)
        ur = cv2.undistortPoints(rc, K2, D2, R=R2, P=P2)
        dy = np.abs(ul[:, 0, 1] - ur[:, 0, 1])
        all_dy.extend(float(v) for v in dy)
        pair_stats.append({"id": pair_id, "median_abs_dy_px": float(np.median(dy)),
                           "max_abs_dy_px": float(np.max(dy))})
        if preview_source is None or np.median(dy) < preview_source[0]:
            preview_source = (float(np.median(dy)), pair_id, left, right)

    values = np.asarray(all_dy)
    result = {
        "point_count": int(values.size),
        "median_abs_dy_px": float(np.median(values)),
        "p95_abs_dy_px": float(np.percentile(values, 95)),
        "max_abs_dy_px": float(np.max(values)),
        "pass": bool(np.percentile(values, 95) <= 2.5),
        "pairs": pair_stats,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    _, pair_id, left, right = preview_source
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_r = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    view = cv2.cvtColor(np.hstack((left_r, right_r)), cv2.COLOR_GRAY2BGR)
    for y in range(25, size[1], 50):
        cv2.line(view, (0, y), (view.shape[1] - 1, y), (0, 180, 0), 1)
    cv2.putText(view, f"rectified pair {pair_id}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(args.preview, view)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
