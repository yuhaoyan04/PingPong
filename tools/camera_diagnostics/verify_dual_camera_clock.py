"""Produce auditable clock-alignment evidence for two DVSLume cameras.

Each camera is owned by a separate process because the official Python binding
is synchronous and not thread-safe.  The final PASS additionally requires a
common optical signal (preferably irregular bright LED pulses) in both views.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from check_dual_camera_sync import estimate_lag, get_sync_tool_type


def parse_args():
    parser = argparse.ArgumentParser(description="DVSLume 双机时钟同步验收")
    parser.add_argument("--master", required=True, help="主相机序列号")
    parser.add_argument("--slave", required=True, help="从相机序列号")
    parser.add_argument("--duration", type=float, default=10.0, help="每路传感器时间采集秒数")
    parser.add_argument("--batch-us", type=int, default=5_000)
    parser.add_argument("--bin-us", type=int, default=100, help="事件率时间桶，默认 100 us")
    parser.add_argument("--max-lag-us", type=int, default=5_000)
    parser.add_argument("--offset-limit-us", type=int, default=500)
    parser.add_argument("--drift-limit-us", type=int, default=200)
    parser.add_argument("--min-correlation", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, default=Path("clock_sync_evidence"))
    return parser.parse_args()


def grow_counts(counts: np.ndarray, needed: int) -> np.ndarray:
    if needed <= len(counts):
        return counts
    return np.pad(counts, (0, max(needed - len(counts), len(counts) or 1024)))


def camera_worker(serial, mode, gate, ready, started, result_queue, config):
    camera = None
    camera_started = False
    try:
        from dvsense_driver.camera_manager import DvsCameraManager

        manager = DvsCameraManager()
        manager.update_cameras()
        camera = manager.open_camera(serial)
        sync_type = get_sync_tool_type()
        tool = camera.get_tool(sync_type)
        if not tool.set_param("mode", mode):
            raise RuntimeError(f"无法设置 {mode}")
        ok, actual_mode = tool.get_param("mode")
        if not ok or str(actual_mode) != mode:
            raise RuntimeError(f"模式回读错误: {actual_mode}")
        camera.set_batch_events_time(config["batch_us"])
        ready.set()
        if not gate.wait(60):
            raise TimeoutError("等待启动命令超时")

        camera.start()
        camera_started = True
        started.set()

        counts = np.zeros(1024, dtype=np.int64)
        batch_host_ns = []
        batch_first_us = []
        batch_last_us = []
        batch_events = []
        first_ts = None
        capture_end = None
        stop_end = None
        total_events = 0
        latest_sensor_ts = None
        while stop_end is None or latest_sensor_ts < stop_end:
            events = camera.get_next_batch()
            host_ns = time.perf_counter_ns()
            if events.size == 0:
                continue
            timestamps = events["timestamp"].astype(np.int64, copy=False)
            latest_sensor_ts = int(timestamps[-1])
            if first_ts is None:
                first_ts = int(timestamps.min())
                capture_end = first_ts + config["duration_us"]
                stop_end = capture_end + config.get("hold_clock_us", 0)
            timestamps = timestamps[timestamps <= capture_end]
            if timestamps.size == 0:
                continue
            first = int(timestamps[0])
            last = int(timestamps[-1])
            indices = timestamps // config["bin_us"]
            counts = grow_counts(counts, int(indices.max()) + 1)
            local = np.bincount(indices, minlength=int(indices.max()) + 1)
            counts[: len(local)] += local
            total_events += len(events)
            batch_host_ns.append(host_ns)
            batch_first_us.append(first)
            batch_last_us.append(last)
            batch_events.append(len(events))

        artifact = Path(config["artifact"])
        np.savez_compressed(
            artifact,
            serial=serial,
            mode=mode,
            bin_us=config["bin_us"],
            counts=counts,
            batch_host_ns=np.asarray(batch_host_ns, dtype=np.int64),
            batch_first_us=np.asarray(batch_first_us, dtype=np.int64),
            batch_last_us=np.asarray(batch_last_us, dtype=np.int64),
            batch_events=np.asarray(batch_events, dtype=np.int64),
        )
        result_queue.put(
            {
                "status": "ok",
                "serial": serial,
                "mode": mode,
                "artifact": str(artifact),
                "first_timestamp_us": first_ts,
                "last_timestamp_us": int(batch_last_us[-1]),
                "total_events": total_events,
                "batch_count": len(batch_last_us),
            }
        )
    except Exception as exc:
        result_queue.put({"status": "error", "serial": serial, "mode": mode, "error": repr(exc)})
        ready.set()
        started.set()
    finally:
        if camera_started:
            try:
                camera.stop()
            except Exception:
                pass


def wait_event(event, process, result_queue, timeout_s, label):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if event.wait(0.1):
            return
        if not process.is_alive():
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                message = {"error": "进程异常退出且没有错误消息"}
            raise RuntimeError(f"{label}: {message}")
    raise TimeoutError(f"{label}超时")


def align_counts(master_counts, slave_counts):
    size = max(len(master_counts), len(slave_counts))
    master = np.pad(master_counts.astype(float), (0, size - len(master_counts)))
    slave = np.pad(slave_counts.astype(float), (0, size - len(slave_counts)))
    active = (master + slave) > 0
    if not np.any(active):
        raise RuntimeError("没有采集到事件")
    lo = max(0, int(np.flatnonzero(active)[0]) - 10)
    hi = min(size, int(np.flatnonzero(active)[-1]) + 11)
    return np.log1p(master[lo:hi]), np.log1p(slave[lo:hi]), lo


def analyze(master_result, slave_result, args, run_dir):
    with np.load(master_result["artifact"]) as data:
        master_counts = data["counts"].copy()
    with np.load(slave_result["artifact"]) as data:
        slave_counts = data["counts"].copy()

    master, slave, first_bin = align_counts(master_counts, slave_counts)
    overall = estimate_lag(master, slave, args.bin_us, args.max_lag_us)
    midpoint = len(master) // 2
    first = estimate_lag(master[:midpoint], slave[:midpoint], args.bin_us, args.max_lag_us)
    second = estimate_lag(master[midpoint:], slave[midpoint:], args.bin_us, args.max_lag_us)

    first_event_delta = slave_result["first_timestamp_us"] - master_result["first_timestamp_us"]
    master_span = master_result["last_timestamp_us"] - master_result["first_timestamp_us"]
    slave_span = slave_result["last_timestamp_us"] - slave_result["first_timestamp_us"]
    span_delta = slave_span - master_span
    lag_drift = second.lag_us - first.lag_us

    streams_ok = master_result["total_events"] > 0 and slave_result["total_events"] > 0
    common_signal_ok = np.isfinite(overall.correlation) and overall.correlation >= args.min_correlation
    offset_ok = abs(overall.lag_us) <= args.offset_limit_us
    drift_ok = abs(lag_drift) <= args.drift_limit_us
    passed = bool(streams_ok and common_signal_ok and offset_ok and drift_ok)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "sdk_modes": {"master": "MASTER", "slave": "SLAVE"},
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "master": master_result,
        "slave": slave_result,
        "measurements": {
            "first_event_delta_slave_minus_master_us": first_event_delta,
            "capture_span_delta_slave_minus_master_us": span_delta,
            "correlation_lag_slave_minus_master_us": overall.lag_us,
            "correlation": overall.correlation,
            "first_half_lag_us": first.lag_us,
            "second_half_lag_us": second.lag_us,
            "lag_drift_us": lag_drift,
        },
        "acceptance": {
            "both_streams": streams_ok,
            "common_signal_detected": bool(common_signal_ok),
            "offset_within_limit": bool(offset_ok),
            "drift_within_limit": bool(drift_ok),
            "passed": passed,
        },
    }
    report_path = run_dir / "clock_sync_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n时钟同步验收结果")
    print(f"  首事件时间戳差（从-主）: {first_event_delta:+d} us")
    print(f"  等传感器时长的跨度差:    {span_delta:+d} us")
    print(f"  共同信号估计偏移（从-主）: {overall.lag_us:+d} us")
    print(f"  共同信号相关系数:          {overall.correlation:.3f}")
    print(f"  前/后半段偏移:             {first.lag_us:+d} / {second.lag_us:+d} us")
    print(f"  偏移变化:                  {lag_drift:+d} us")
    print(f"  证据报告: {report_path.resolve()}")
    if passed:
        print("\n最终结论: PASS，两路时钟通过当前验收阈值。")
        return 0
    if not common_signal_ok:
        print("\n最终结论: PENDING。两路均已锁定出流，但没有检测到足够强的共同光学信号，不能证明时间戳零点。")
        return 2
    print("\n最终结论: FAIL。检测到共同信号，但偏移或漂移超过验收阈值。")
    return 1


def main():
    args = parse_args()
    if args.master == args.slave:
        raise SystemExit("主从相机序列号不能相同")
    if min(args.duration, args.batch_us, args.bin_us, args.offset_limit_us, args.drift_limit_us) <= 0:
        raise SystemExit("时间参数必须大于 0")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    config_base = {
        "duration_us": int(args.duration * 1_000_000),
        "batch_us": args.batch_us,
        "bin_us": args.bin_us,
    }

    processes = {}
    controls = {}
    try:
        for role, serial, mode in (("slave", args.slave, "SLAVE"), ("master", args.master, "MASTER")):
            gate, ready, started = ctx.Event(), ctx.Event(), ctx.Event()
            config = config_base | {
                "artifact": str((run_dir / f"{role}_evidence.npz").resolve()),
                "hold_clock_us": 1_000_000 if role == "master" else 0,
            }
            process = ctx.Process(
                target=camera_worker,
                args=(serial, mode, gate, ready, started, result_queue, config),
                name=f"dvs-{role}",
            )
            process.start()
            processes[role] = process
            controls[role] = (gate, ready, started)
            wait_event(ready, process, result_queue, 120, f"打开{role}")

        controls["slave"][0].set()
        wait_event(controls["slave"][2], processes["slave"], result_queue, 15, "启动从机")
        time.sleep(0.2)
        controls["master"][0].set()
        wait_event(controls["master"][2], processes["master"], result_queue, 15, "启动主机")
        print(f"主从机已启动。采集 {args.duration:g} 秒传感器时间，请制造两机共同可见的不规则 LED 闪烁。")

        results = {}
        deadline = time.monotonic() + max(60, args.duration * 10)
        while len(results) < 2 and time.monotonic() < deadline:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if message["status"] != "ok":
                raise RuntimeError(f"{message['mode']} / {message['serial']}: {message['error']}")
            results[message["mode"]] = message
        if len(results) != 2:
            raise TimeoutError("采集进程未在限定时间内完成")
        for process in processes.values():
            process.join(timeout=10)
        return analyze(results["MASTER"], results["SLAVE"], args, run_dir)
    finally:
        # Keep the master's clock alive until the slave has stopped.
        for role in ("slave", "master"):
            process = processes.get(role)
            if process is None:
                continue
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(3)
