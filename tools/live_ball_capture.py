"""双路事件球实验：可视倒计时、同步预览和自动 RAW 录制。"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.acquisition import (
    configure_boolean_parameter, configure_event_rate_control,
    configure_trail_filter)


MAX_RENDER_EVENTS = 500_000


def render(events, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 72, np.uint8)
    if events is None or len(events) == 0:
        return image
    if len(events) > MAX_RENDER_EVENTS:
        events = events[::int(np.ceil(len(events) / MAX_RENDER_EVENTS))]
    x = events["x"].astype(np.intp); y = events["y"].astype(np.intp)
    p = events["polarity"].astype(bool)
    valid = (x < width) & (y < height)
    image[y[valid & ~p], x[valid & ~p]] = (255, 80, 30)
    image[y[valid & p], x[valid & p]] = (30, 240, 255)
    return image


def update_activity(activity: np.ndarray, events, decay: float = 0.82) -> np.ndarray:
    """Accumulate several event batches so a fast, sparse ball stays visible."""
    activity *= decay
    if events is not None and len(events):
        x = events["x"].astype(np.intp); y = events["y"].astype(np.intp)
        valid = (x < activity.shape[1]) & (y < activity.shape[0])
        np.add.at(activity, (y[valid], x[valid]), 1.0)
    return activity


def activity_bgr(activity: np.ndarray) -> np.ndarray:
    level = np.percentile(activity[activity > 0], 99.5) if np.any(activity > 0) else 1.0
    gray = np.clip(activity * (255.0 / max(float(level), 1.0)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def text(image, line: str, y: int, color=(80, 255, 80), scale=0.8):
    cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
                cv2.LINE_AA)


def polarity_counts(events) -> tuple[int, int, float]:
    if events is None or len(events) == 0:
        return 0, 0, 0.0
    on = int(np.count_nonzero(events["polarity"].astype(bool)))
    off = int(len(events) - on)
    return on, off, min(on, off) / max(on, off, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="DSP01F1FFF59")
    ap.add_argument("--slave", default="DSP01F1FFF3E")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--manual-start", action="store_true",
                    help="Keep preview open and start recording only after SPACE")
    ap.add_argument("--manual-stop", action="store_true",
                    help="Stop and save on SPACE instead of using the duration limit")
    ap.add_argument("--batch-us", type=int, default=5000)
    ap.add_argument("--preview-decay", type=float, default=0.82,
                    help="Preview-only activity persistence in [0, 1); RAW events are unchanged")
    ap.add_argument("--max-event-rate", type=int, default=10,
                    help="ERC limit in MEv/s per camera")
    ap.add_argument("--event-rate-control", action="store_true",
                    help="Enable event-dropping ERC; disabled by default for spin texture")
    ap.add_argument("--bias-diff-on", type=int, default=0,
                    help="DVSLume ON contrast-bias offset; official default is 0")
    ap.add_argument("--bias-diff-off", type=int, default=0,
                    help="DVSLume OFF contrast-bias offset; official default is 0")
    ap.add_argument("--all-bias-zero", action="store_true",
                    help="Set and verify all six DVSLume IMX636 bias offsets to zero")
    ap.add_argument("--trail-filter", action="store_true",
                    help="Enable TRAIL filter; disabled by default because it deletes texture")
    ap.add_argument("--trail-threshold-ms", type=int, default=1)
    ap.add_argument("--output-dir", default="output/ball_trials")
    ap.add_argument("--safe-margin-px", type=int, default=0,
                    help="Draw a full-ball safety rectangle this many pixels from every sensor edge")
    args = ap.parse_args()

    # Hardware SDK is only required when opening actual cameras.
    from dvsense_driver.camera_manager import DvsCameraManager
    from dvsense_driver.hal.camera_tool import ToolType
    if not 0.0 <= args.preview_decay < 1.0:
        raise SystemExit("--preview-decay must be in [0, 1)")
    if args.batch_us <= 0 or args.duration <= 0:
        raise SystemExit("--batch-us and --duration must be positive")
    if args.all_bias_zero and (args.bias_diff_on != 0 or args.bias_diff_off != 0):
        raise SystemExit("--all-bias-zero cannot be combined with nonzero contrast bias")

    manager = DvsCameraManager(); manager.update_cameras()
    found = {d.serial for d in manager.get_camera_descs()}
    missing = {args.master, args.slave} - found
    if missing:
        raise SystemExit(f"未发现相机: {sorted(missing)}")
    master = manager.open_camera(args.master); slave = manager.open_camera(args.slave)
    sync_type = getattr(ToolType, "SYNC", getattr(ToolType, "TOOL_SYNC", ToolType(2)))
    hardware_readback = {}
    for camera, mode in ((master, "MASTER"), (slave, "SLAVE")):
        tool = camera.get_tool(sync_type)
        if not tool.set_param("mode", mode):
            raise RuntimeError(f"不能设置 {mode}")
        ok, actual = tool.get_param("mode")
        if not ok or str(actual) != mode:
            raise RuntimeError(f"同步模式回读失败: {actual}")
        readback = {"sync_mode": str(actual), "batch_us": int(args.batch_us)}
        erc = camera.get_tool(ToolType.EVENT_RATE_CONTROL)
        readback["event_rate_control"] = configure_event_rate_control(
            erc, args.event_rate_control, args.max_event_rate, mode)
        bias = camera.get_tool(ToolType.BIAS)
        bias_values = (("bias_diff_on", args.bias_diff_on), ("bias_diff_off", args.bias_diff_off))
        if args.all_bias_zero:
            bias_values = tuple((name, 0) for name in (
                "bias_fo", "bias_hpf", "bias_diff_on", "bias_diff",
                "bias_diff_off", "bias_refr"))
        for name, value in bias_values:
            if not bias.set_param(name, value):
                raise RuntimeError(f"不能设置 {mode} {name}={value}")
            ok, actual = bias.get_param(name)
            if not ok or int(actual) != value:
                raise RuntimeError(f"{mode} {name} 回读失败: {actual}")
            readback[name] = int(actual)
        trail = camera.get_tool(ToolType.EVENT_TRAIL_FILTER)
        readback["trail_filter"] = configure_trail_filter(
            trail, args.trail_filter, args.trail_threshold_ms, mode)
        readback["anti_flicker_enabled"] = configure_boolean_parameter(
            camera.get_tool(ToolType.ANTI_FLICKER), "enable", False,
            f"{mode} Anti Flicker")
        camera.set_batch_events_time(args.batch_us)
        hardware_readback[mode.lower()] = readback

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    master_raw = (run_dir / f"master_{args.master}.raw").resolve()
    slave_raw = (run_dir / f"slave_{args.slave}.raw").resolve()
    recording = False; completed = False
    preview_started = None; record_started = None; record_ended = None
    manual_trigger = False
    first_ts = {"master": None, "slave": None}; last_ts = {"master": None, "slave": None}
    capture_stats = {
        name: {"events": 0, "batches": 0, "saturated_batches": 0,
               "nonmonotonic_batches": 0, "maximum_batch_events": 0}
        for name in ("master", "slave")
    }
    previous_batch_last = {"master": None, "slave": None}
    master_activity = slave_activity = None
    try:
        slave.start(); time.sleep(0.15); master.start()
        while True:
            se = slave.get_next_batch(); me = master.get_next_batch()
            now = time.monotonic()
            if preview_started is None:
                preview_started = now
            for name, events in (("master", me), ("slave", se)):
                if events is not None and len(events):
                    lo, hi = int(events["timestamp"].min()), int(events["timestamp"].max())
                    first_ts[name] = lo if first_ts[name] is None else min(first_ts[name], lo)
                    last_ts[name] = hi
                    if recording:
                        stats = capture_stats[name]
                        stats["events"] += int(len(events)); stats["batches"] += 1
                        stats["maximum_batch_events"] = max(stats["maximum_batch_events"], int(len(events)))
                        if args.event_rate_control:
                            expected_cap = args.max_event_rate * args.batch_us
                            stats["saturated_batches"] += int(
                                len(events) >= 0.98 * expected_cap)
                        prior = previous_batch_last[name]
                        stats["nonmonotonic_batches"] += int(prior is not None and lo <= prior)
                        previous_batch_last[name] = hi
            should_start = (manual_trigger if args.manual_start
                            else now - preview_started >= args.countdown)
            if not recording and record_started is None and should_start:
                slave.start_recording(str(slave_raw)); master.start_recording(str(master_raw))
                recording = True; record_started = now
                print(f"REC START {run_dir.resolve()}")
            w, h = master.get_width(), master.get_height()
            if master_activity is None:
                master_activity = np.zeros((h, w), np.float32)
                slave_activity = np.zeros((h, w), np.float32)
            update_activity(master_activity, me, args.preview_decay)
            update_activity(slave_activity, se, args.preview_decay)
            left = cv2.addWeighted(render(me, w, h), 0.45, activity_bgr(master_activity), 0.75, 0)
            right = cv2.addWeighted(render(se, w, h), 0.45, activity_bgr(slave_activity), 0.75, 0)
            text(left, f"MASTER {args.master[-3:]} events={0 if me is None else len(me)}", 34)
            text(right, f"SLAVE {args.slave[-3:]} events={0 if se is None else len(se)}", 34)
            master_on, master_off, master_balance = polarity_counts(me)
            slave_on, slave_off, slave_balance = polarity_counts(se)
            if record_started is None:
                if args.manual_start:
                    status, color = "READY  press SPACE to record", (0, 220, 255)
                else:
                    remain = max(0.0, args.countdown - (now - preview_started))
                    status, color = f"PREPARE  recording in {remain:.1f}s", (0, 220, 255)
            elif args.manual_stop:
                status, color = f"REC  {now - record_started:.1f}s  SPACE to stop", (0, 0, 255)
            else:
                remain = max(0.0, args.duration - (now - record_started))
                status, color = f"REC  {remain:.1f}s", (0, 0, 255)
            text(left, status, 72, color, 0.9); text(right, status, 72, color, 0.9)
            text(left, "Keep target inside BOTH views", 108, (255, 255, 255), 0.65)
            text(right, "Keep target inside BOTH views", 108, (255, 255, 255), 0.65)
            text(left, f"ON/OFF={master_on}/{master_off} balance={master_balance:.2f}",
                 142, (255, 255, 255), 0.58)
            text(right, f"ON/OFF={slave_on}/{slave_off} balance={slave_balance:.2f}",
                 142, (255, 255, 255), 0.58)
            erc_status = (f"ERC ON {args.max_event_rate} MEv/s"
                          if args.event_rate_control else "ERC OFF (full texture)")
            text(left, erc_status, 174, (0, 220, 255), 0.55)
            text(right, erc_status, 174, (0, 220, 255), 0.55)
            text(left, f"preview={args.batch_us/1000:g}ms decay={args.preview_decay:g}; RAW us timestamps",
                 204, (255, 255, 255), 0.48)
            text(right, f"preview={args.batch_us/1000:g}ms decay={args.preview_decay:g}; RAW us timestamps",
                 204, (255, 255, 255), 0.48)
            if args.safe_margin_px > 0:
                margin = min(int(args.safe_margin_px), w // 2 - 1, h // 2 - 1)
                for panel in (left, right):
                    cv2.rectangle(panel, (margin, margin), (w - margin, h - margin),
                                  (0, 255, 0), 2, cv2.LINE_AA)
                    text(panel, "Keep the COMPLETE ball inside the green box",
                         h - 24, (0, 255, 0), 0.62)
            view = cv2.resize(np.hstack((left, right)), None, fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_AREA)
            cv2.imshow("DVSLume DUAL CAPTURE | SPACE record | Q quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and args.manual_start and record_started is None:
                manual_trigger = True
            elif key == ord(" ") and args.manual_stop and recording:
                slave.stop_recording(); master.stop_recording()
                record_ended = time.monotonic()
                recording = False; completed = True
                break
            if key in (ord("q"), 27):
                break
            if recording and not args.manual_stop and now - record_started >= args.duration:
                slave.stop_recording(); master.stop_recording()
                record_ended = time.monotonic()
                recording = False; completed = True
                break
    finally:
        if recording:
            try: slave.stop_recording()
            except Exception: pass
            try: master.stop_recording()
            except Exception: pass
            record_ended = time.monotonic()
        for camera in (slave, master):
            try: camera.stop()
            except Exception: pass
        cv2.destroyAllWindows()
    metadata = {
        "completed": completed, "duration_s": None if args.manual_stop else args.duration,
        "actual_duration_s": (record_ended - record_started
                              if record_started is not None and record_ended is not None else None),
        "manual_stop": args.manual_stop,
        "batch_us": args.batch_us,
        "preview_decay": args.preview_decay,
        "event_rate_control_enabled": args.event_rate_control,
        "max_event_rate_mevs": args.max_event_rate if args.event_rate_control else None,
        "bias_diff_on": args.bias_diff_on, "bias_diff_off": args.bias_diff_off,
        "all_bias_zero_requested": args.all_bias_zero,
        "trail_filter_enabled": args.trail_filter,
        "trail_threshold_ms": args.trail_threshold_ms if args.trail_filter else None,
        "spin_capture_config_valid": bool(
            not args.trail_filter and not args.event_rate_control),
        "hardware_readback": hardware_readback,
        "master_raw": str(master_raw), "slave_raw": str(slave_raw),
        "first_timestamp_us": first_ts, "last_timestamp_us": last_ts,
        "capture_stats": capture_stats,
    }
    for stats in capture_stats.values():
        batches = max(stats["batches"], 1)
        stats["saturation_fraction"] = (
            stats["saturated_batches"] / batches if args.event_rate_control else None)
    (run_dir / "capture.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
