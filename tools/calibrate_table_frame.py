"""Build an auditable master-camera to table coordinate transform."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_stereo import corners
from pingpong_event_pipeline.stereo import StereoCalibration


def blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 8; params.maxThreshold = 255
    params.filterByArea = True; params.minArea = 12; params.maxArea = 20000
    params.filterByCircularity = True; params.minCircularity = .12
    params.filterByConvexity = False; params.filterByInertia = False
    params.filterByColor = True; params.blobColor = 255
    return cv2.SimpleBlobDetector_create(params)


def numbered_overlay(image_path: Path, points: np.ndarray, output: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    for index, point in enumerate(points.reshape(-1, 2)):
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(image, xy, 8, (0, 255, 0), 2)
        cv2.putText(image, str(index), (xy[0] + 8, xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 80, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), image)


def build_table_frame(points_master: np.ndarray, origin_index: int,
                      x_index: int, y_index: int):
    points = np.asarray(points_master, np.float64).reshape(-1, 3)
    measured_origin = points[origin_index]
    raw_x = points[x_index] - measured_origin
    raw_y = points[y_index] - measured_origin
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    z_axis = vh[-1]
    if np.dot(z_axis, np.cross(raw_x, raw_y)) < 0:
        z_axis *= -1
    # Put the table origin on the best-fit plane rather than retaining the
    # depth noise of one LED. Axis directions remain defined by the chosen
    # physical neighboring LEDs.
    origin = measured_origin - z_axis * np.dot(measured_origin - centroid, z_axis)
    x_axis = raw_x - z_axis * np.dot(raw_x, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    if np.dot(y_axis, raw_y) < 0:
        y_axis *= -1; z_axis *= -1
    R = np.vstack((x_axis, y_axis, z_axis))
    return R, origin


def main() -> None:
    ap = argparse.ArgumentParser(description="用平放在桌面的 LED 板标定球桌坐标")
    ap.add_argument("--master-image", required=True); ap.add_argument("--slave-image", required=True)
    ap.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    ap.add_argument("--cols", type=int, default=3); ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--origin-index", type=int); ap.add_argument("--x-index", type=int)
    ap.add_argument("--y-index", type=int); ap.add_argument("--output", default="calibration/table_frame.npz")
    args = ap.parse_args()
    master_path, slave_path = Path(args.master_image), Path(args.slave_image)
    detector = blob_detector(); pattern = (args.cols, args.rows)
    _, left = corners(master_path, pattern, "led-circles", detector)
    _, right = corners(slave_path, pattern, "led-circles", detector)
    if left is None or right is None:
        raise SystemExit("两路图像未同时识别到完整 LED GRID")
    if np.dot(left[-1, 0] - left[0, 0], right[-1, 0] - right[0, 0]) < 0:
        right = right[::-1].copy()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    numbered_overlay(master_path, left, output.with_name(output.stem + "_master_numbered.png"))
    numbered_overlay(slave_path, right, output.with_name(output.stem + "_slave_numbered.png"))
    calibration = StereoCalibration(args.calibration)
    points, errors = [], []
    for lp, rp in zip(left[:, 0], right[:, 0]):
        point, error = calibration.triangulate(lp, rp); points.append(point); errors.append(error)
    points = np.asarray(points); errors = np.asarray(errors)
    if args.origin_index is None or args.x_index is None or args.y_index is None:
        preview = {
            "status": "indices_required",
            "instruction": "查看 numbered 图片；原点选靠近己方右角，+X 选朝对方邻点，+Y 选朝左邻点。",
            "master_numbered": str(output.with_name(output.stem + "_master_numbered.png").resolve()),
            "slave_numbered": str(output.with_name(output.stem + "_slave_numbered.png").resolve()),
            "reprojection_p95_px": float(np.percentile(errors, 95)),
        }
        output.with_suffix(".json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(preview, ensure_ascii=False, indent=2)); return
    indices = (args.origin_index, args.x_index, args.y_index)
    if any(index < 0 or index >= len(points) for index in indices) or len(set(indices)) < 3:
        raise SystemExit("origin/x/y 索引必须是三个不同的有效灯编号")
    R, origin = build_table_frame(points, *indices)
    table_points = (R @ (points - origin).T).T
    plane_rms = float(np.sqrt(np.mean(table_points[:, 2] ** 2)))
    normal_toward_camera = float(np.dot(R[2], -origin)) > 0
    neighbor_distances = []
    xy = table_points[:, :2]
    for i in range(len(xy)):
        distances = np.linalg.norm(xy[i + 1:] - xy[i], axis=1)
        neighbor_distances.extend(distances[(distances > .025) & (distances < .075)])
    spacing_median = float(np.median(neighbor_distances)) if neighbor_distances else float("nan")
    passed = bool(plane_rms <= .003 and np.percentile(errors, 95) <= 2.5 and
                  normal_toward_camera and .04 <= spacing_median <= .06)
    np.savez(output, R_table_from_master=R, origin_master_m=origin,
             points_master_m=points, points_table_m=table_points,
             origin_index=args.origin_index, x_index=args.x_index, y_index=args.y_index)
    report = {"status": "table_frame_valid" if passed else "needs_review",
              "plane_rms_mm": plane_rms * 1000,
              "reprojection_p95_px": float(np.percentile(errors, 95)),
              "neighbor_spacing_median_mm": spacing_median * 1000,
              "normal_points_toward_camera": normal_toward_camera,
              "origin_index": args.origin_index, "x_index": args.x_index,
              "y_index": args.y_index, "output": str(output.resolve())}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("球桌外参未通过质量门槛，请检查编号方向或重新采集")


if __name__ == "__main__":
    main()
