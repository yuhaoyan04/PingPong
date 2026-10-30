"""Live dual-camera trajectory display with lossless RAW capture for spin audit."""
from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dvsense_driver.camera_manager import DvsCameraManager
from dvsense_driver.hal.camera_tool import ToolType

from pingpong_event_pipeline.config import load_config
from pingpong_event_pipeline.acquisition import (
    configure_boolean_parameter, configure_event_rate_control,
    configure_trail_filter)
from pingpong_event_pipeline.events import EVENT_DTYPE, RawWindowReader
from pingpong_event_pipeline.quality import marker_structure_metrics
from pingpong_event_pipeline.realtime import CausalStereoTracker
from pingpong_event_pipeline.template_spin_channel import StereoTemplateSpinChannel
from live_ball_capture import polarity_counts, render, text


FIELDS = ["timestamp_us", "master_x", "master_y", "slave_x", "slave_y",
          "measured_x_m", "measured_y_m", "measured_z_m", "x_m", "y_m", "z_m",
          "vx_mps", "vy_mps", "vz_mps", "speed_mps", "reprojection_error_px",
          "spin_master_candidates", "spin_slave_candidates", "spin_markers_3d",
          "spin_pose_valid", "spin_pose_mode", "spin_template_inliers",
          "spin_pose_residual_chord", "spin_rpm", "omega_x_rad_s", "omega_y_rad_s",
          "omega_z_rad_s", "spin_axis_x", "spin_axis_y", "spin_axis_z",
          "spin_axis_resultant"]


def configure_camera(camera, mode: str, batch_us: int, event_rate: int,
                     event_rate_control: bool = False,
                     bias_diff_on: int = 0, bias_diff_off: int = 0,
                     trail_filter: bool = False, trail_threshold_ms: int = 1) -> dict:
    sync_type = getattr(ToolType, "SYNC", getattr(ToolType, "TOOL_SYNC", ToolType(2)))
    sync = camera.get_tool(sync_type)
    if not sync.set_param("mode", mode):
        raise RuntimeError(f"不能设置同步模式 {mode}")
    ok, actual = sync.get_param("mode")
    if not ok or str(actual) != mode:
        raise RuntimeError(f"同步模式回读失败：{actual}")
    readback = {"sync_mode": str(actual), "batch_us": int(batch_us)}
    erc = camera.get_tool(ToolType.EVENT_RATE_CONTROL)
    readback["event_rate_control"] = configure_event_rate_control(
        erc, event_rate_control, event_rate, mode)
    readback["anti_flicker_enabled"] = configure_boolean_parameter(
        camera.get_tool(ToolType.ANTI_FLICKER), "enable", False,
        f"{mode} Anti Flicker")
    bias = camera.get_tool(ToolType.BIAS)
    for name, value in (("bias_diff_on", bias_diff_on),
                        ("bias_diff_off", bias_diff_off)):
        if not bias.set_param(name, value):
            raise RuntimeError(f"不能设置 {mode} {name}={value}")
        ok, actual = bias.get_param(name)
        if not ok or int(actual) != value:
            raise RuntimeError(f"{mode} {name} 回读失败: {actual}")
        readback[name] = int(actual)
    trail = camera.get_tool(ToolType.EVENT_TRAIL_FILTER)
    readback["trail_filter"] = configure_trail_filter(
        trail, trail_filter, trail_threshold_ms, mode)
    camera.set_batch_events_time(batch_us)
    return readback


def result_row(result, spin_result=None):
    row = {field: "" for field in FIELDS}; row["timestamp_us"] = result.timestamp_us
    for prefix, track in (("master", result.master_track), ("slave", result.slave_track)):
        if track is not None:
            row[prefix + "_x"], row[prefix + "_y"] = track.x, track.y
    if result.measured_position_m is not None:
        row.update(dict(zip(("measured_x_m", "measured_y_m", "measured_z_m"),
                            result.measured_position_m)))
        row["reprojection_error_px"] = result.reprojection_error_px
    if result.filtered_position_m is not None:
        row.update(dict(zip(("x_m", "y_m", "z_m"), result.filtered_position_m)))
        row.update(dict(zip(("vx_mps", "vy_mps", "vz_mps"), result.velocity_mps)))
        row["speed_mps"] = result.speed_mps
    if spin_result is not None:
        row["spin_master_candidates"] = spin_result.master_candidates
        row["spin_slave_candidates"] = spin_result.slave_candidates
        row["spin_markers_3d"] = spin_result.triangulated_markers
        if spin_result.stream is not None:
            stream = spin_result.stream
            row["spin_pose_valid"] = int(stream.pose_valid)
            row["spin_pose_mode"] = stream.pose_mode
            row["spin_template_inliers"] = stream.template_inliers
            row["spin_pose_residual_chord"] = (stream.pose_residual_chord
                                                if np.isfinite(stream.pose_residual_chord) else "")
            if stream.rpm is not None:
                row["spin_rpm"] = stream.rpm
                omega = stream.spin.omega_rad_s
                axis = stream.spin.axis
                row.update(dict(zip(("omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s"), omega)))
                row.update(dict(zip(("spin_axis_x", "spin_axis_y", "spin_axis_z"), axis)))
                row["spin_axis_resultant"] = stream.spin.axis_resultant
    return row


def candidate_windows(states_csv: Path, master_raw: Path, slave_raw: Path,
                      maximum_gap_us: int = 30_000, padding_us: int = 50_000):
    try:
        data = np.atleast_1d(np.genfromtxt(
            states_csv, delimiter=",", names=True, encoding="utf-8-sig", filling_values=np.nan))
        valid = np.isfinite(data["measured_x_m"])
        timestamps = data["timestamp_us"][valid].astype(np.int64)
        if len(timestamps) < 5:
            return []
        groups = np.split(timestamps, np.flatnonzero(np.diff(timestamps) > maximum_gap_us) + 1)
        origin = max(RawWindowReader(master_raw).info.start_us,
                     RawWindowReader(slave_raw).info.start_us)
        windows = []
        for group in groups:
            if len(group) < 5:
                continue
            start = max(origin, int(group[0]) - padding_us)
            end = int(group[-1]) + padding_us
            windows.append({"start_offset_s": (start - origin) * 1e-6,
                            "duration_s": (end - start) * 1e-6,
                            "online_stereo_points": len(group)})
        return windows
    except Exception as error:
        return [{"error": str(error)}]


def select_activity_peaks(edges_us, master_counts, slave_counts, origin_us,
                          maximum_windows=4, separation_us=500_000,
                          pre_peak_us=40_000, post_peak_us=300_000):
    """Select separated periods that are simultaneously active in both cameras."""
    left = np.asarray(master_counts, np.float64)
    right = np.asarray(slave_counts, np.float64)
    score = np.sqrt(left * right)
    selected = []
    for index in np.argsort(score)[::-1]:
        center = (int(edges_us[index]) + int(edges_us[index + 1])) // 2
        if score[index] < 1000 or any(abs(center - old) < separation_us for old in selected):
            continue
        selected.append(center)
        if len(selected) >= maximum_windows:
            break
    selected.sort()
    result = []
    for center in selected:
        start = max(int(origin_us), center - pre_peak_us)
        result.append({"start_offset_s": (start - int(origin_us)) * 1e-6,
                       "duration_s": (pre_peak_us + post_peak_us) * 1e-6,
                       "source": "raw_stereo_activity"})
    return result


def raw_activity_windows(master_raw: Path, slave_raw: Path, roi,
                         maximum_windows=4, bin_us=20_000):
    """Find fallback flight windows from lossless RAW when online tracking fell behind."""
    readers = [RawWindowReader(master_raw), RawWindowReader(slave_raw)]
    origin = max(reader.info.start_us for reader in readers)
    end = min(reader.info.end_us for reader in readers)
    edges = np.arange(origin, end + bin_us, bin_us, dtype=np.int64)
    if len(edges) < 2:
        return []
    x0, y0, x1, y1 = map(int, roi)
    counts = []
    for reader in readers:
        camera_counts = np.zeros(len(edges) - 1, np.int64)
        # Bounded reads avoid loading a long high-rate recording into RAM at once.
        for first in range(0, len(camera_counts), 500):
            last = min(first + 500, len(camera_counts))
            events = reader.window(int(edges[first]), int(edges[last]))
            inside = ((events["x"] >= x0) & (events["x"] < x1) &
                      (events["y"] >= y0) & (events["y"] < y1))
            camera_counts[first:last] = np.histogram(
                events["timestamp"][inside].astype(np.int64), edges[first:last + 1])[0]
        counts.append(camera_counts)
    return select_activity_peaks(
        edges, counts[0], counts[1], origin, maximum_windows=maximum_windows)


def main() -> None:
    ap = argparse.ArgumentParser(description="双路 DVSLume 在线轨迹/球速显示和 RAW 录制")
    ap.add_argument("--config", default="config/realtime_trajectory.yaml")
    ap.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--manual-start", action="store_true")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--output-dir", default="output/live_measurements")
    ap.add_argument("--max-event-rate", type=int, default=10)
    ap.add_argument("--event-rate-control", action="store_true",
                    help="Enable event-dropping ERC; disabled by default for spin texture")
    ap.add_argument("--bias-diff-on", type=int, default=0)
    ap.add_argument("--bias-diff-off", type=int, default=0)
    ap.add_argument("--trail-filter", action="store_true",
                    help="Enable texture-deleting TRAIL filter; off by default for spin")
    ap.add_argument("--trail-threshold-ms", type=int, default=1)
    ap.add_argument("--queue-batches", type=int, default=128)
    ap.add_argument("--display-fps", type=float, default=15.0)
    ap.add_argument("--auto-process", action="store_true",
                    help="Run trajectory/marker preflight after capture; never outputs RPM")
    ap.add_argument("--max-auto-windows", type=int, default=8)
    ap.add_argument("--table-frame", help="Optional calibration/table_frame.npz")
    ap.add_argument("--no-measure-preview", action="store_true")
    ap.add_argument("--spin-template", help="validated physical_ball_template.npz")
    ap.add_argument("--spin-template-report", help="accepted physical_ball_template.json")
    args = ap.parse_args()
    if bool(args.spin_template) != bool(args.spin_template_report):
        ap.error("--spin-template and --spin-template-report must be provided together")
    if args.spin_template and (args.trail_filter or args.event_rate_control):
        ap.error("validated live spin requires Trail Filter and ERC to remain disabled")
    cfg = load_config(args.config); camera_cfg = cfg["cameras"]
    batch_us = int(cfg["events"]["step_us"])
    manager = DvsCameraManager(); manager.update_cameras()
    found = {desc.serial for desc in manager.get_camera_descs()}
    missing = {camera_cfg["master_serial"], camera_cfg["slave_serial"]} - found
    if missing:
        raise SystemExit(f"未发现相机: {sorted(missing)}")
    master = manager.open_camera(camera_cfg["master_serial"])
    slave = manager.open_camera(camera_cfg["slave_serial"])
    master_readback = configure_camera(
        master, "MASTER", batch_us, args.max_event_rate, args.event_rate_control,
        args.bias_diff_on, args.bias_diff_off,
        args.trail_filter, args.trail_threshold_ms)
    slave_readback = configure_camera(
        slave, "SLAVE", batch_us, args.max_event_rate, args.event_rate_control,
        args.bias_diff_on, args.bias_diff_off,
        args.trail_filter, args.trail_threshold_ms)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    master_raw = (run_dir / f"master_{camera_cfg['master_serial']}.raw").resolve()
    slave_raw = (run_dir / f"slave_{camera_cfg['slave_serial']}.raw").resolve()
    states_csv = run_dir / "live_trajectory.csv"
    tracker = CausalStereoTracker(cfg, args.calibration)
    spin_channel = (StereoTemplateSpinChannel(
        args.calibration, args.spin_template, args.spin_template_report,
        cfg.get("template_spin")) if args.spin_template else None)
    work = queue.Queue(maxsize=max(4, args.queue_batches)); stop = threading.Event()
    shared = {"latest": None, "latest_spin": None, "processed": 0, "stereo": 0,
              "spin_outputs": 0, "wall_s": 0.0, "dropped": 0, "latency_s": 0.0}

    def tracking_worker():
        with states_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
            while not stop.is_set() or not work.empty():
                try:
                    timestamp, me, se, recorded, enqueued = work.get(timeout=.1)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                result = tracker.process(timestamp, me, se)
                spin_result = None
                if spin_channel is not None:
                    center = (result.measured_position_m if result.measured_position_m is not None
                              else result.filtered_position_m)
                    spin_result = spin_channel.process(
                        timestamp, me, se, result.master_track, result.slave_track, center)
                shared["wall_s"] += time.perf_counter() - started
                shared["processed"] += 1; shared["stereo"] += int(result.measured_position_m is not None)
                shared["spin_outputs"] += int(spin_result is not None and spin_result.rpm is not None)
                shared["latency_s"] = time.monotonic() - enqueued
                shared["latest"] = result; shared["latest_spin"] = spin_result
                if recorded:
                    writer.writerow(result_row(result, spin_result)); handle.flush()
                work.task_done()

    worker = threading.Thread(target=tracking_worker, daemon=True); worker.start()
    recording = False; record_started = None; preview_started = None; manual_trigger = False
    last_display = 0.0; completed = False
    try:
        slave.start(); time.sleep(.15); master.start()
        while True:
            se = slave.get_next_batch(); me = master.get_next_batch(); now = time.monotonic()
            if preview_started is None: preview_started = now
            should_start = manual_trigger if args.manual_start else now - preview_started >= args.countdown
            if not recording and record_started is None and should_start:
                slave.start_recording(str(slave_raw)); master.start_recording(str(master_raw))
                recording = True; record_started = now; print(f"REC START {run_dir.resolve()}")
            event_arrays = [events for events in (me, se) if events is not None and len(events)]
            if event_arrays:
                # Use the newest timestamp so every event in this paired SDK
                # delivery has a non-negative age in the causal time surface.
                timestamp = max(int(events["timestamp"].max()) for events in event_arrays)
                master_copy = me.copy() if me is not None else np.empty(0, dtype=EVENT_DTYPE)
                slave_copy = se.copy() if se is not None else np.empty(0, dtype=EVENT_DTYPE)
                item = (timestamp, master_copy, slave_copy, recording, now)
                if work.full():
                    try: work.get_nowait(); work.task_done(); shared["dropped"] += 1
                    except queue.Empty: pass
                work.put_nowait(item)
            if now - last_display >= 1.0 / max(args.display_fps, 1.0):
                left = render(me, master.get_width(), master.get_height())
                right = render(se, slave.get_width(), slave.get_height())
                latest = shared["latest"]
                if latest:
                    for image, track in ((left, latest.master_track), (right, latest.slave_track)):
                        if track:
                            cv2.circle(image, (round(track.x), round(track.y)),
                                       max(2, round(track.radius)), (0, 255, 0), 2)
                    speed = latest.speed_mps
                    status = "TRACK %.2f m/s" % speed if speed is not None else "SEARCHING BALL"
                    latest_spin = shared["latest_spin"]
                    if latest_spin is not None and latest_spin.rpm is not None:
                        status += "  SPIN %.0f RPM" % latest_spin.rpm
                else: status = "WARMING UP"
                realtime_factor = shared["processed"] * batch_us * 1e-6 / max(shared["wall_s"], 1e-9)
                health = "REALTIME" if realtime_factor >= 1 and shared["dropped"] == 0 else "BEHIND"
                for image in (left, right):
                    text(image, status, 36); text(image, f"{health} factor={realtime_factor:.2f}x drop={shared['dropped']}",
                                                  72, (0, 255, 0) if health == "REALTIME" else (0, 100, 255), .65)
                    text(image, f"latency={shared['latency_s']*1000:.0f} ms", 104, (255, 255, 255), .6)
                master_on, master_off, master_balance = polarity_counts(me)
                slave_on, slave_off, slave_balance = polarity_counts(se)
                text(left, f"ON/OFF={master_on}/{master_off} balance={master_balance:.2f}",
                     136, (255, 255, 255), .55)
                text(right, f"ON/OFF={slave_on}/{slave_off} balance={slave_balance:.2f}",
                     136, (255, 255, 255), .55)
                for image, events, track in ((left, me, latest.master_track if latest else None),
                                             (right, se, latest.slave_track if latest else None)):
                    if track is None:
                        marker_line = "BALL texture: no track"
                        marker_color = (0, 100, 255)
                    else:
                        metrics = marker_structure_metrics(
                            events, track.timestamp_us, track.point, (track.vx, track.vy),
                            track.radius)
                        marker_line = (f"BALL n={metrics['events']} bal={metrics['polarity_balance']:.2f} "
                                       f"peak={metrics['density_peak_to_mean']:.1f}")
                        marker_color = ((0, 255, 0) if metrics["events"] >= 20
                                        and metrics["polarity_balance"] >= .25
                                        and metrics["density_peak_to_mean"] >= 6 else (0, 100, 255))
                    text(image, marker_line, 168, marker_color, .55)
                cv2.imshow("DVSLume LIVE MEASURE | SPACE start | Q stop",
                           cv2.resize(np.hstack((left, right)), None, fx=.5, fy=.5))
                last_display = now
            key = cv2.waitKey(1) & 0xff
            if key == ord(" ") and args.manual_start and record_started is None: manual_trigger = True
            if key in (ord("q"), 27): break
            if recording and now - record_started >= args.duration:
                slave.stop_recording(); master.stop_recording(); recording = False; completed = True; break
    finally:
        if recording:
            try: slave.stop_recording()
            except Exception: pass
            try: master.stop_recording()
            except Exception: pass
        for camera in (slave, master):
            try: camera.stop()
            except Exception: pass
        stop.set(); worker.join(timeout=10); cv2.destroyAllWindows()
    report = {"completed": completed, "raw": {"master": str(master_raw), "slave": str(slave_raw)},
              "live_states_csv": str(states_csv.resolve()), **{key: value for key, value in shared.items()
              if key not in ("latest", "latest_spin")}}
    report["acquisition_config"] = {
        "batch_us": batch_us,
        "event_rate_control_enabled": args.event_rate_control,
        "max_event_rate_mevs": args.max_event_rate if args.event_rate_control else None,
        "bias_diff_on": args.bias_diff_on,
        "bias_diff_off": args.bias_diff_off,
        "trail_filter_enabled": args.trail_filter,
        "trail_threshold_ms": args.trail_threshold_ms if args.trail_filter else None,
        "spin_capture_config_valid": bool(
            not args.trail_filter and not args.event_rate_control),
        "hardware_readback": {
            "master": master_readback,
            "slave": slave_readback,
        },
    }
    report["realtime_factor"] = (shared["processed"] * batch_us * 1e-6 /
                                 max(shared["wall_s"], 1e-9))
    report["live_template_spin"] = {
        "enabled": spin_channel is not None,
        "template": str(Path(args.spin_template).resolve()) if args.spin_template else None,
        "template_report": (str(Path(args.spin_template_report).resolve())
                            if args.spin_template_report else None),
        "accepted_outputs": shared["spin_outputs"],
    }
    report["candidate_windows"] = candidate_windows(states_csv, master_raw, slave_raw)
    if not any("start_offset_s" in window for window in report["candidate_windows"]):
        try:
            report["candidate_windows"] = raw_activity_windows(
                master_raw, slave_raw, cfg["detection"]["search_roi"],
                maximum_windows=max(1, args.max_auto_windows))
        except Exception as error:
            report["raw_activity_fallback_error"] = str(error)
    root = Path(__file__).resolve().parents[1]
    preflight_jobs = []
    for index, window in enumerate(report["candidate_windows"][:max(1, args.max_auto_windows)]):
        if "start_offset_s" not in window:
            continue
        output = run_dir / ("spin_preflight_" + str(index + 1).zfill(2))
        command = [sys.executable, str(root / "tools" / "preflight_spin_capture.py"),
                   "--master-raw", str(master_raw), "--slave-raw", str(slave_raw),
                   "--start-offset-s", f"{window['start_offset_s']:.6f}",
                   "--duration-s", f"{window['duration_s']:.6f}",
                   "--output-dir", str(output)]
        preflight_jobs.append((command, output))
    report["spin_preflight_commands"] = [subprocess.list2cmdline(command)
                                         for command, _ in preflight_jobs]
    report["spin_solver_allowed"] = bool(spin_channel is not None and shared["spin_outputs"] > 0)
    report["spin_solver_gate"] = (
        "Validated physical template plus accepted causal rolling pose window required.")
    report_path = run_dir / "live_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.auto_process:
        report["automatic_spin_preflights"] = []
        for command, output in preflight_jobs:
            log_path = output.with_name(output.name + "_run.log")
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed_process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, text=True)
            manifest = output / "spin_preflight.json"
            summary = None
            if manifest.exists():
                try:
                    summary = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            report["automatic_spin_preflights"].append({
                "output_dir": str(output.resolve()),
                "return_code": completed_process.returncode,
                "runtime_seconds": time.perf_counter() - started,
                "manifest": str(manifest.resolve()) if manifest.exists() else None,
                "status": summary.get("status") if summary else "preflight_failed",
                "spin_solver_allowed": False,
                "log": str(log_path.resolve()),
            })
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
