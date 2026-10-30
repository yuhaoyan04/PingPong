"""双路实时视野调整与成对标定帧采集。Q 退出，S 保存当前帧。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from dvsense_driver.camera_manager import DvsCameraManager
from dvsense_driver.hal.camera_tool import ToolType


MAX_RENDER_EVENTS = 500_000


def sampled_events(events):
    """Bound preview memory when lighting creates millions of events per batch."""
    if events is None or len(events) <= MAX_RENDER_EVENTS:
        return events
    step = int(np.ceil(len(events) / MAX_RENDER_EVENTS))
    return events[::step]


def event_image(events, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 80, np.uint8)
    if events is None or len(events) == 0:
        return image
    events = sampled_events(events)
    x = events["x"].astype(np.intp)
    y = events["y"].astype(np.intp)
    p = events["polarity"].astype(bool)
    valid = (x < width) & (y < height)
    image[y[valid & ~p], x[valid & ~p]] = (255, 80, 30)
    image[y[valid & p], x[valid & p]] = (30, 230, 255)
    return image


def calibration_image(events, width: int, height: int) -> np.ndarray:
    """Polarity image with symmetric contrast, suitable for corner extraction."""
    image = np.full((height, width), 127, np.uint8)
    if events is None or len(events) == 0:
        return image
    events = sampled_events(events)
    x = events["x"].astype(np.intp); y = events["y"].astype(np.intp)
    p = events["polarity"].astype(bool); valid = (x < width) & (y < height)
    image[y[valid & ~p], x[valid & ~p]] = 0
    image[y[valid & p], x[valid & p]] = 255
    return image


def event_count_image(events, width: int, height: int) -> np.ndarray:
    """Count events per pixel without letting event polarity cancel LED activity."""
    if events is None or len(events) == 0:
        return np.zeros((height, width), np.float32)
    # Uniform sampling preserves spatial density and bounds large USB bursts.
    events = sampled_events(events)
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    linear = y[valid] * width + x[valid]
    return np.bincount(linear, minlength=width * height).reshape(height, width).astype(np.float32)


def accumulated_activity(accumulator: np.ndarray) -> np.ndarray:
    """Convert a temporal event-count surface into a robust 8-bit LED image."""
    smooth = cv2.GaussianBlur(accumulator, (0, 0), 1.6)
    nonzero = smooth[smooth > 0]
    if nonzero.size == 0:
        return np.zeros(accumulator.shape, np.uint8)
    # Ignore isolated low-count background events, then clip hot pixels.
    floor = float(np.percentile(nonzero, 55.0))
    ceiling = float(np.percentile(nonzero, 99.85))
    if ceiling <= floor:
        ceiling = floor + 1.0
    scaled = np.clip((smooth - floor) * (255.0 / (ceiling - floor)), 0, 255)
    return scaled.astype(np.uint8)


def polarity_activity(gray: np.ndarray, positive: bool) -> np.ndarray:
    """Extract one motion edge polarity so a moving LED stays a compact disk."""
    delta = gray.astype(np.int16) - 127
    signal = np.maximum(delta if positive else -delta, 0).astype(np.uint8)
    return cv2.GaussianBlur(signal, (5, 5), 0)


def make_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 8
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 12
    params.maxArea = 20000
    params.filterByCircularity = True
    params.minCircularity = 0.12
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255
    return cv2.SimpleBlobDetector_create(params)


def find_led_grid(activity: np.ndarray, detector):
    # Half resolution is enough for the 3x4 board and prevents OpenCV's
    # clustering stage from monopolising the GUI on a noisy full-HD event map.
    work = cv2.resize(activity, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    keypoints = detector.detect(work)
    if not 8 <= len(keypoints) <= 80:
        return False, None, activity, len(keypoints)
    ok, centers = cv2.findCirclesGrid(
        work, (3, 4), flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
        blobDetector=detector)
    if ok:
        centers *= 2.0
    return ok, centers, activity, len(keypoints)


def detect_led_grid(gray: np.ndarray, accumulated: np.ndarray, detector):
    """Try compact instantaneous ON/OFF disks before the motion accumulator."""
    attempts = (polarity_activity(gray, True), polarity_activity(gray, False), accumulated)
    best = (False, None, attempts[0], 0)
    best_distance = float("inf")
    for activity in attempts:
        result = find_led_grid(activity, detector)
        if result[0]:
            return result
        distance = abs(result[3] - 12)
        if distance < best_distance:
            best, best_distance = result, distance
    return best


def label(image, text: str, events) -> None:
    count = 0 if events is None else len(events)
    cv2.putText(image, f"{text}  events={count}", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80), 2)
    cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (80, 80, 80), 1)
    cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (80, 80, 80), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="DSP01F1FFF59")
    ap.add_argument("--slave", default="DSP01F1FFF3E")
    ap.add_argument("--batch-us", type=int, default=50000,
                    help="视野/标定累积时长；默认 50ms，追踪阶段使用配置中的 5ms")
    ap.add_argument("--scale", type=float, default=0.55)
    ap.add_argument("--save-dir", default="calibration/captures")
    ap.add_argument("--auto-debug-after", type=float, default=0.0,
                    help="启动若干秒后自动保存一次诊断图；0 表示关闭")
    ap.add_argument("--auto-save-interval", type=float, default=0.8,
                    help="两路同时识别后自动保存的最短间隔；0 表示关闭")
    ap.add_argument("--max-auto-pairs", type=int, default=30,
                    help="自动采集的标定帧对总数上限")
    args = ap.parse_args()

    manager = DvsCameraManager()
    manager.update_cameras()
    found = {d.serial for d in manager.get_camera_descs()}
    missing = {args.master, args.slave} - found
    if missing:
        raise SystemExit(f"未发现相机: {sorted(missing)}；当前发现: {sorted(found)}")
    master = manager.open_camera(args.master)
    slave = manager.open_camera(args.slave)
    sync_type = getattr(ToolType, "SYNC", getattr(ToolType, "TOOL_SYNC", ToolType(2)))
    for camera, mode in ((master, "MASTER"), (slave, "SLAVE")):
        tool = camera.get_tool(sync_type)
        if not tool.set_param("mode", mode):
            raise RuntimeError(f"无法把 {camera} 设置为 {mode}")
        ok, actual = tool.get_param("mode")
        if not ok or str(actual) != mode:
            raise RuntimeError(f"同步模式回读失败：期望 {mode}，实际 {actual}")
    # LED 标定板依赖周期亮灭事件；AFK 会把它当作频闪噪声抑制。
    for camera in (master, slave):
        camera.get_tool(ToolType.ANTI_FLICKER).set_param("enable", False)
        camera.get_tool(ToolType.EVENT_TRAIL_FILTER).set_param("enable", False)
        erc = camera.get_tool(ToolType.EVENT_RATE_CONTROL)
        erc.set_param("max_event_rate", 20)
        erc.set_param("enable", True)
    for camera in (slave, master):
        camera.set_batch_events_time(args.batch_us)
    # 已验证的同步启动顺序不可交换。
    slave.start()
    time.sleep(0.15)
    master.start()
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    pair_index = len(list(output.glob("master_*.png")))
    blob_detector = make_blob_detector()
    master_acc = np.zeros((master.get_height(), master.get_width()), np.float32)
    slave_acc = np.zeros((slave.get_height(), slave.get_width()), np.float32)
    preview_started = time.monotonic()
    auto_debug_saved = False
    frame_index = 0
    left_ok = right_ok = False
    left_centers = right_centers = None
    left_blobs = right_blobs = 0
    capture_left = capture_right = None
    last_auto_save = 0.0
    print("实时视野已打开：缓慢挥动带格纹/文字的球或标定板；S 保存成对帧，Q 退出。")
    try:
        while True:
            slave_events = slave.get_next_batch()
            master_events = master.get_next_batch()
            w, h = master.get_width(), master.get_height()
            left, right = event_image(master_events, w, h), event_image(slave_events, w, h)
            left_cal = calibration_image(master_events, w, h)
            right_cal = calibration_image(slave_events, w, h)
            # About 10 successive batches remain visible. This spans LED blinking
            # phases while old motion/noise fades away.
            master_acc *= 0.88
            slave_acc *= 0.88
            master_acc += event_count_image(master_events, w, h)
            slave_acc += event_count_image(slave_events, w, h)
            left_activity = accumulated_activity(master_acc)
            right_activity = accumulated_activity(slave_acc)
            frame_index += 1
            detection_now = frame_index % 5 == 1
            if detection_now:
                left_ok, left_centers, detected_left, left_blobs = detect_led_grid(
                    left_cal, left_activity, blob_detector)
                right_ok, right_centers, detected_right, right_blobs = detect_led_grid(
                    right_cal, right_activity, blob_detector)
                if left_ok:
                    left_activity = detected_left
                    capture_left = detected_left.copy()
                if right_ok:
                    right_activity = detected_right
                    capture_right = detected_right.copy()
            label(left, "MASTER F59", master_events)
            label(right, "SLAVE F3E", slave_events)
            cv2.putText(left, "LED GRID: OK" if left_ok else f"LED GRID: searching blobs={left_blobs}",
                        (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if left_ok else (0, 180, 255), 2)
            cv2.putText(right, "LED GRID: OK" if right_ok else f"LED GRID: searching blobs={right_blobs}",
                        (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if right_ok else (0, 180, 255), 2)
            if left_ok:
                cv2.drawChessboardCorners(left, (3, 4), left_centers, True)
            if right_ok:
                cv2.drawChessboardCorners(right, (3, 4), right_centers, True)
            view = cv2.resize(np.hstack((left, right)), None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_AREA)
            cv2.imshow("DVSLume stereo FOV | S save | Q quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                if not (left_ok and right_ok and capture_left is not None and capture_right is not None):
                    print("未保存：两路必须同时显示 LED GRID: OK")
                    continue
                pair_index += 1
                cv2.imwrite(str(output / f"master_{pair_index:03d}.png"), capture_left)
                cv2.imwrite(str(output / f"slave_{pair_index:03d}.png"), capture_right)
                print(f"已保存标定帧对 {pair_index:03d}")
            now = time.monotonic()
            if (detection_now and left_ok and right_ok and
                    args.auto_save_interval > 0 and pair_index < args.max_auto_pairs and
                    now - last_auto_save >= args.auto_save_interval):
                pair_index += 1
                cv2.imwrite(str(output / f"master_{pair_index:03d}.png"), capture_left)
                cv2.imwrite(str(output / f"slave_{pair_index:03d}.png"), capture_right)
                last_auto_save = now
                print(f"自动保存标定帧对 {pair_index:03d}/{args.max_auto_pairs}")
            if key == ord("d"):
                stamp = int(time.time())
                cv2.imwrite(str(output / f"debug_master_{stamp}.png"), left_cal)
                cv2.imwrite(str(output / f"debug_slave_{stamp}.png"), right_cal)
                cv2.imwrite(str(output / f"debug_master_activity_{stamp}.png"), left_activity)
                cv2.imwrite(str(output / f"debug_slave_activity_{stamp}.png"), right_activity)
                print(f"已保存诊断图 {stamp}（无需 GRID OK）")
            if (args.auto_debug_after > 0 and not auto_debug_saved and
                    time.monotonic() - preview_started >= args.auto_debug_after):
                stamp = int(time.time())
                cv2.imwrite(str(output / f"debug_master_{stamp}.png"), left_cal)
                cv2.imwrite(str(output / f"debug_slave_{stamp}.png"), right_cal)
                cv2.imwrite(str(output / f"debug_master_activity_{stamp}.png"), left_activity)
                cv2.imwrite(str(output / f"debug_slave_activity_{stamp}.png"), right_activity)
                print(f"已自动保存诊断图 {stamp}（无需 GRID OK）")
                auto_debug_saved = True
    finally:
        for camera in (master, slave):
            try:
                camera.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
