"""Record two synchronized DVSLume raw streams, then verify timestamps offline."""

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


def camera_recorder(
    serial, mode, raw_path, start_gate, started, record_gate, stop_gate, messages
):
    camera = None
    streaming = False
    recording = False
    try:
        from dvsense_driver.camera_manager import DvsCameraManager

        manager = DvsCameraManager()
        manager.update_cameras()
        camera = manager.open_camera(serial)
        tool = camera.get_tool(get_sync_tool_type())
        if not tool.set_param("mode", mode):
            raise RuntimeError(f"不能设置 {mode}")
        ok, actual = tool.get_param("mode")
        if not ok or str(actual) != mode:
            raise RuntimeError(f"模式回读为 {actual}")
        messages.put({"kind": "ready", "serial": serial, "mode": mode})
        if not start_gate.wait(120):
            raise TimeoutError("等待启动超时")
        camera.start()
        streaming = True
        started.set()
        if not record_gate.wait(120):
            raise TimeoutError("等待录制命令超时")
        camera.start_recording(str(raw_path))
        recording = True
        messages.put({"kind": "recording", "serial": serial, "mode": mode})
        if not stop_gate.wait(300):
            raise TimeoutError("等待停止超时")
        camera.stop_recording()
        recording = False
        camera.stop()
        streaming = False
        messages.put({"kind": "done", "serial": serial, "mode": mode, "raw": str(raw_path)})
    except Exception as exc:
        messages.put({"kind": "error", "serial": serial, "mode": mode, "error": repr(exc)})
        started.set()
    finally:
        if recording:
            try:
                camera.stop_recording()
            except Exception:
                pass
        if streaming:
            try:
                camera.stop()
            except Exception:
                pass


def get_message(messages, expected_kind, expected_mode, timeout):
    deadline = time.monotonic() + timeout
    deferred = []
    try:
        while time.monotonic() < deadline:
            try:
                message = messages.get(timeout=0.5)
            except queue.Empty:
                continue
            if message["kind"] == "error":
                raise RuntimeError(f"{message['mode']} {message['serial']}: {message['error']}")
            if message["kind"] == expected_kind and message["mode"] == expected_mode:
                return message
            deferred.append(message)
        raise TimeoutError(f"等待 {expected_mode} {expected_kind} 超时")
    finally:
        for message in deferred:
            messages.put(message)


def read_raw_counts(raw_path: Path, bin_us: int, chunk_us: int = 20_000):
    from dvsense_driver.raw_file_reader import RawFileReader

    reader = RawFileReader(str(raw_path))
    if not reader.load_file():
        raise RuntimeError(f"不能载入 {raw_path}")
    ok_start, file_start = reader.get_start_timestamp()
    ok_end, file_end = reader.get_end_timestamp()
    if not ok_start or not ok_end:
        raise RuntimeError(f"不能读取 {raw_path} 的时间范围")

    counts = np.zeros(max(1024, int(file_end // bin_us) + 2), dtype=np.int64)
    first_event = None
    last_event = None
    total = 0
    while not reader.reached_end_of_events():
        events = reader.get_n_time_events(chunk_us)
        if events.size:
            timestamps = events["timestamp"].astype(np.int64, copy=False)
            if first_event is None:
                first_event = int(timestamps.min())
            last_event = int(timestamps.max())
            indices = timestamps // bin_us
            if int(indices.max()) >= len(counts):
                counts = np.pad(counts, (0, int(indices.max()) + 2 - len(counts)))
            local = np.bincount(indices, minlength=int(indices.max()) + 1)
            counts[: len(local)] += local
            total += len(events)
        if reader.get_current_pos_timestamp() + chunk_us >= file_end:
            break
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    return {
        "file": str(raw_path.resolve()),
        "file_start_us": int(file_start),
        "file_end_us": int(file_end),
        "first_event_us": first_event,
        "last_event_us": last_event,
        "events": total,
        "counts": counts,
    }


def analyze(master, slave, args, run_dir):
    if master["first_event_us"] is None or slave["first_event_us"] is None:
        report = {
            "created_at": datetime.now().astimezone().isoformat(),
            "method": "single-process SDK raw recording; offline timestamp analysis",
            "master": {key: value for key, value in master.items() if key != "counts"},
            "slave": {key: value for key, value in slave.items() if key != "counts"},
            "passed": False,
            "reason": "one or both RAW streams contain no events",
        }
        report_path = run_dir / "clock_sync_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n最终结论: FAIL，至少一路 RAW 没有事件。报告: {report_path.resolve()}")
        return 1
    size = max(len(master["counts"]), len(slave["counts"]))
    m = np.pad(master["counts"].astype(float), (0, size - len(master["counts"])))
    s = np.pad(slave["counts"].astype(float), (0, size - len(slave["counts"])))
    active = (m + s) > 0
    indices = np.flatnonzero(active)
    lo, hi = max(0, int(indices[0]) - 10), min(size, int(indices[-1]) + 11)
    m, s = np.log1p(m[lo:hi]), np.log1p(s[lo:hi])
    overall = estimate_lag(m, s, args.bin_us, args.max_lag_us)
    mid = len(m) // 2
    first = estimate_lag(m[:mid], s[:mid], args.bin_us, args.max_lag_us)
    second = estimate_lag(m[mid:], s[mid:], args.bin_us, args.max_lag_us)

    first_delta = slave["first_event_us"] - master["first_event_us"]
    span_m = master["last_event_us"] - master["first_event_us"]
    span_s = slave["last_event_us"] - slave["first_event_us"]
    drift = second.lag_us - first.lag_us
    common = bool(np.isfinite(overall.correlation) and overall.correlation >= args.min_correlation)
    passed = common and abs(overall.lag_us) <= args.offset_limit_us and abs(drift) <= args.drift_limit_us

    def public(data):
        return {key: value for key, value in data.items() if key != "counts"}

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "single-process SDK raw recording; offline timestamp analysis",
        "master": public(master),
        "slave": public(slave),
        "measurements": {
            "first_event_delta_slave_minus_master_us": first_delta,
            "event_span_delta_slave_minus_master_us": span_s - span_m,
            "common_signal_lag_slave_minus_master_us": overall.lag_us,
            "common_signal_correlation": overall.correlation,
            "first_half_lag_us": first.lag_us,
            "second_half_lag_us": second.lag_us,
            "lag_drift_us": drift,
        },
        "limits": {
            "offset_us": args.offset_limit_us,
            "drift_us": args.drift_limit_us,
            "minimum_correlation": args.min_correlation,
        },
        "bin_us": args.bin_us,
        "interpretation": "Coarse global event-rate correlation only; this does not establish optical timing accuracy, event integrity, or a true zero offset.",
        "passed": passed,
    }
    report_path = run_dir / "clock_sync_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n离线证据分析")
    print(f"  主/从事件数: {master['events']:,} / {slave['events']:,}")
    print(f"  首事件差（从-主）: {first_delta:+d} us")
    print(f"  事件时间跨度差:     {span_s - span_m:+d} us")
    print(f"  共同信号偏移:       {overall.lag_us:+d} us")
    print(f"  相关系数:           {overall.correlation:.3f}")
    print(f"  前/后半段偏移:      {first.lag_us:+d} / {second.lag_us:+d} us")
    print(f"  偏移变化:           {drift:+d} us")
    print(f"  报告: {report_path.resolve()}")
    if passed:
        print("\n粗筛: PASS，满足事件率相关性阈值；不代表光学同步精度或事件完整性已验收。")
        return 0
    if not common:
        print("\n最终结论: PENDING，硬件主从出流正常，但没有检测到足够强的两机共同信号。")
        return 2
    print("\n最终结论: FAIL，共同信号的偏移或漂移超过阈值。")
    return 1


def parse_args():
    parser = argparse.ArgumentParser(description="双 DVSLume 原始流同步验收")
    parser.add_argument("--master", required=True)
    parser.add_argument("--slave", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--max-lag-us", type=int, default=5_000)
    parser.add_argument("--offset-limit-us", type=int, default=500)
    parser.add_argument("--drift-limit-us", type=int, default=200)
    parser.add_argument("--min-correlation", type=float, default=0.50)
    parser.add_argument(
        "--max-event-rate",
        type=int,
        default=20,
        help="每台相机的 ERC 上限（MEv/s）；设为 0 时不启用 ERC",
    )
    parser.add_argument("--contrast-bias", type=int, default=20)
    parser.add_argument("--trail-filter-ms", type=int, default=10)
    parser.add_argument("--anti-flicker", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("clock_sync_raw_evidence"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    from dvsense_driver.camera_manager import DvsCameraManager

    manager = DvsCameraManager()
    manager.update_cameras()
    descs = manager.get_camera_descs()
    serials = {desc.serial for desc in descs}
    if args.master not in serials or args.slave not in serials:
        raise RuntimeError(f"没有同时发现指定的两台相机；当前发现: {sorted(serials)}")
    master_camera = manager.open_camera(args.master)
    slave_camera = manager.open_camera(args.slave)
    sync_type = get_sync_tool_type()
    master_tool = master_camera.get_tool(sync_type)
    slave_tool = slave_camera.get_tool(sync_type)
    if not master_tool.set_param("mode", "MASTER"):
        raise RuntimeError("不能设置主机模式")
    if not slave_tool.set_param("mode", "SLAVE"):
        raise RuntimeError("不能设置从机模式")

    if args.max_event_rate > 0:
        from dvsense_driver.hal.camera_tool import ToolType

        for role, camera in (("主机", master_camera), ("从机", slave_camera)):
            erc = camera.get_tool(ToolType.EVENT_RATE_CONTROL)
            if not erc.set_param("max_event_rate", args.max_event_rate):
                raise RuntimeError(f"不能设置{role} ERC 事件率上限")
            if not erc.set_param("enable", True):
                raise RuntimeError(f"不能启用{role} ERC")
        print(f"两台相机 ERC 已限制为 {args.max_event_rate} MEv/s。")

    from dvsense_driver.hal.camera_tool import ToolType

    for role, camera in (("主机", master_camera), ("从机", slave_camera)):
        bias = camera.get_tool(ToolType.BIAS)
        for param in ("bias_diff_on", "bias_diff_off"):
            if not bias.set_param(param, args.contrast_bias):
                raise RuntimeError(f"不能设置{role} {param}")
        trail = camera.get_tool(ToolType.EVENT_TRAIL_FILTER)
        if args.trail_filter_ms > 0:
            if not trail.set_param("threshold", args.trail_filter_ms):
                raise RuntimeError(f"不能设置{role} Event Trail Filter 阈值")
            if not trail.set_param("type", "TRAIL"):
                raise RuntimeError(f"不能设置{role} Event Trail Filter 类型")
            if not trail.set_param("enable", True):
                raise RuntimeError(f"不能启用{role} Event Trail Filter")
        elif not trail.set_param("enable", False):
            raise RuntimeError(f"不能关闭{role} Event Trail Filter")
    trail_description = f"{args.trail_filter_ms} ms" if args.trail_filter_ms > 0 else "关闭"
    print(f"两台相机对比度 bias 为 {args.contrast_bias:+d}，Event Trail Filter {trail_description}。")

    if args.anti_flicker:
        for role, camera in (("主机", master_camera), ("从机", slave_camera)):
            anti_flicker = camera.get_tool(ToolType.ANTI_FLICKER)
            for param, value in (
                ("low_frequency", 50),
                ("high_frequency", 520),
                ("fliter_mode", "Band cut"),
                ("enable", True),
            ):
                if not anti_flicker.set_param(param, value):
                    raise RuntimeError(f"不能设置{role} Anti Flicker 参数 {param}")
        print("两台相机 Anti Flicker 已启用（50–520 Hz Band cut）。")

    master_raw = (run_dir / f"master_{args.master}.raw").resolve()
    slave_raw = (run_dir / f"slave_{args.slave}.raw").resolve()
    master_started = slave_started = False
    master_recording = slave_recording = False
    try:
        slave_camera.start()
        slave_started = True
        time.sleep(0.2)
        master_camera.start()
        master_started = True
        time.sleep(0.2)
        slave_camera.start_recording(str(slave_raw))
        slave_recording = True
        master_camera.start_recording(str(master_raw))
        master_recording = True
        print(f"正在录制 {args.duration:g} 秒；请让大白纸在两机重叠视野内快速扫过。")
        time.sleep(args.duration)
        slave_camera.stop_recording()
        slave_recording = False
        master_camera.stop_recording()
        master_recording = False
    finally:
        if slave_recording:
            slave_camera.stop_recording()
        if master_recording:
            master_camera.stop_recording()
        if slave_started:
            slave_camera.stop()
        if master_started:
            master_camera.stop()

    print("录制完成，正在离线读取硬件时间戳……")
    master = read_raw_counts(master_raw, args.bin_us)
    slave = read_raw_counts(slave_raw, args.bin_us)
    return analyze(master, slave, args, run_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(3)
