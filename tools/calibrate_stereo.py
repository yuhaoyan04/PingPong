"""从 master_NNN/slave_NNN 图像对完成棋盘格双目标定。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def corners(path: Path, pattern: tuple[int, int], pattern_type: str, detector=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    if pattern_type == "led-circles":
        # Auto-capture already stores the successful single-polarity activity
        # image. Use the exact half-resolution path used by the live detector.
        activity = cv2.resize(image, None, fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_AREA)
        ok, pts = cv2.findCirclesGrid(
            activity, pattern,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
            blobDetector=detector)
        if ok:
            pts *= 2.0
        return image, pts if ok else None
    candidates = [image, cv2.equalizeHist(image), cv2.GaussianBlur(image, (5, 5), 0)]
    for candidate in candidates:
        ok, pts = cv2.findChessboardCornersSB(
            candidate, pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            return image, pts
    return image, None


def view_errors(object_points, image_points, K, D, rvecs, tvecs):
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected = cv2.projectPoints(obj, rvec, tvec, K, D)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=2)))))
    return np.asarray(errors)


def main() -> None:
    ap = argparse.ArgumentParser(description="DVSLume 双目棋盘格标定")
    ap.add_argument("--images", default="calibration/captures")
    ap.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--pattern", choices=("chessboard", "led-circles"), default="chessboard")
    ap.add_argument("--spacing-x-mm", type=float, help="LED 圆心水平间距")
    ap.add_argument("--spacing-y-mm", type=float, help="LED 圆心垂直间距")
    ap.add_argument("--output", default="calibration/stereo_calibration.npz")
    args = ap.parse_args()
    root, pattern = Path(args.images), (args.cols, args.rows)
    masters = {p.stem.split("_", 1)[1]: p for p in root.glob("master_*.png")}
    slaves = {p.stem.split("_", 1)[1]: p for p in root.glob("slave_*.png")}
    ids = sorted(masters.keys() & slaves.keys())
    if args.pattern == "led-circles" and (not args.spacing_x_mm or not args.spacing_y_mm):
        raise SystemExit("LED 圆点标定必须提供 --spacing-x-mm 和 --spacing-y-mm")
    sx = (args.spacing_x_mm if args.pattern == "led-circles" else args.square_mm) / 1000.0
    sy = (args.spacing_y_mm if args.pattern == "led-circles" else args.square_mm) / 1000.0
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    grid = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    obj[:, 0], obj[:, 1] = grid[:, 0] * sx, grid[:, 1] * sy
    detector = None
    if args.pattern == "led-circles":
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 8; params.maxThreshold = 255
        params.filterByArea = True; params.minArea = 12; params.maxArea = 20000
        params.filterByCircularity = True; params.minCircularity = 0.12
        params.filterByConvexity = False; params.filterByInertia = False
        params.filterByColor = True; params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
    object_points, left_points, right_points, valid_ids = [], [], [], []
    size = None
    for pair_id in ids:
        left, lc = corners(masters[pair_id], pattern, args.pattern, detector)
        right, rc = corners(slaves[pair_id], pattern, args.pattern, detector)
        if lc is not None and rc is not None:
            # A symmetric grid has a 180-degree indexing ambiguity. OpenCV can
            # independently reverse one camera's order, which creates valid
            # detections but false stereo correspondences. Match endpoint
            # directions before appending the pair.
            lv = lc[-1, 0] - lc[0, 0]
            rv = rc[-1, 0] - rc[0, 0]
            if float(np.dot(lv, rv)) < 0:
                rc = rc[::-1].copy()
            size = (left.shape[1], left.shape[0])
            object_points.append(obj.copy()); left_points.append(lc); right_points.append(rc)
            valid_ids.append(pair_id)
            print(f"[OK] {pair_id}")
        else:
            print(f"[SKIP] {pair_id}: 两路未同时识别到完整棋盘格")
    if len(object_points) < 12:
        raise SystemExit(f"有效帧对仅 {len(object_points)}，至少需要 12 对（建议 20–30 对覆盖画面四周和不同倾角）")
    # Reject frames whose circle centers came from a smeared accumulator. A few
    # such frames can dominate the intrinsic fit even when most views are sharp.
    rejected = []
    for _ in range(3):
        _, K1, D1, rv1, tv1 = cv2.calibrateCamera(object_points, left_points, size, None, None)
        _, K2, D2, rv2, tv2 = cv2.calibrateCamera(object_points, right_points, size, None, None)
        le = view_errors(object_points, left_points, K1, D1, rv1, tv1)
        re = view_errors(object_points, right_points, K2, D2, rv2, tv2)
        limit_l = max(3.0, float(np.median(le) * 2.5))
        limit_r = max(3.0, float(np.median(re) * 2.5))
        keep = (le <= limit_l) & (re <= limit_r)
        if bool(np.all(keep)) or int(np.count_nonzero(keep)) < 12:
            break
        for i in np.flatnonzero(~keep):
            rejected.append({"id": valid_ids[i], "left_px": float(le[i]), "right_px": float(re[i])})
        object_points = [v for v, ok in zip(object_points, keep) if ok]
        left_points = [v for v, ok in zip(left_points, keep) if ok]
        right_points = [v for v, ok in zip(right_points, keep) if ok]
        valid_ids = [v for v, ok in zip(valid_ids, keep) if ok]
    _, K1, D1, _, _ = cv2.calibrateCamera(object_points, left_points, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(object_points, right_points, size, None, None)
    if rejected:
        print("自动剔除离群帧: " + ", ".join(v["id"] for v in rejected))
    flags = cv2.CALIB_FIX_INTRINSIC
    for _ in range(3):
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            object_points, left_points, right_points, K1, D1, K2, D2, size, flags=flags)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T)
        medians, maxima = [], []
        for lc, rc in zip(left_points, right_points):
            ul = cv2.undistortPoints(lc, K1, D1, R=R1, P=P1)
            ur = cv2.undistortPoints(rc, K2, D2, R=R2, P=P2)
            dy = np.abs(ul[:, 0, 1] - ur[:, 0, 1])
            medians.append(float(np.median(dy))); maxima.append(float(np.max(dy)))
        keep = (np.asarray(medians) <= 2.5) & (np.asarray(maxima) <= 5.0)
        if bool(np.all(keep)) or int(np.count_nonzero(keep)) < 12:
            break
        for i in np.flatnonzero(~keep):
            rejected.append({"id": valid_ids[i], "reason": "epipolar",
                             "median_abs_dy_px": medians[i], "max_abs_dy_px": maxima[i]})
        object_points = [v for v, ok in zip(object_points, keep) if ok]
        left_points = [v for v, ok in zip(left_points, keep) if ok]
        right_points = [v for v, ok in zip(right_points, keep) if ok]
        valid_ids = [v for v, ok in zip(valid_ids, keep) if ok]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, image_size=np.asarray(size), rms=rms,
             valid_pairs=len(object_points), spacing_x_m=sx, spacing_y_m=sy,
             pattern_type=args.pattern)
    baseline = float(np.linalg.norm(T))
    report = {"rms_px": float(rms), "baseline_m": baseline,
              "valid_pairs": len(object_points), "image_size": list(size),
              "used_pair_ids": valid_ids, "rejected_pairs": rejected,
              "pass": bool(rms <= 1.5 and 0.03 <= baseline <= 2.0)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"标定完成: RMS={rms:.3f}px, baseline={baseline:.4f}m, pairs={len(object_points)}")
    print(output.resolve())
    if rms > 1.5:
        print("警告：RMS 偏高，请补拍清晰且覆盖画面边缘的帧后重做。")


if __name__ == "__main__":
    main()
