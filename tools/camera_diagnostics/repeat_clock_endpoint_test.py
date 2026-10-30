"""Repeat short dual RAW recordings to verify hardware timestamp stability."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from check_dual_camera_sync import get_sync_tool_type, set_mode


def raw_range(path):
    from dvsense_driver.raw_file_reader import RawFileReader

    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"不能读取 {path}")
    ok_start, start = reader.get_start_timestamp()
    ok_end, end = reader.get_end_timestamp()
    if not ok_start or not ok_end:
        raise RuntimeError(f"不能读取时间范围 {path}")
    return int(start), int(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--slave", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--limit-us", type=int, default=500)
    parser.add_argument("--packet-period-us", type=int, default=4_000)
    parser.add_argument("--output-dir", type=Path, default=Path("clock_endpoint_evidence"))
    args = parser.parse_args()
    if args.master == args.slave or args.trials < 1 or args.duration <= 0 or args.packet_period_us <= 0:
        parser.error("相机序列号必须不同，次数、时长和包周期必须为正")

    from dvsense_driver.camera_manager import DvsCameraManager
    from dvsense_driver.hal.camera_tool import ToolType

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    manager = DvsCameraManager()
    manager.update_cameras()
    master = manager.open_camera(args.master)
    slave = manager.open_camera(args.slave)
    set_mode(master, get_sync_tool_type(), "MASTER")
    set_mode(slave, get_sync_tool_type(), "SLAVE")
    for camera in (master, slave):
        erc = camera.get_tool(ToolType.EVENT_RATE_CONTROL)
        erc.set_param("max_event_rate", 5)
        erc.set_param("enable", True)
        bias = camera.get_tool(ToolType.BIAS)
        bias.set_param("bias_diff_on", 20)
        bias.set_param("bias_diff_off", 20)
        trail = camera.get_tool(ToolType.EVENT_TRAIL_FILTER)
        trail.set_param("threshold", 10)
        trail.set_param("type", "TRAIL")
        trail.set_param("enable", True)

    rows = []
    slave.start()
    time.sleep(0.2)
    master.start()
    time.sleep(0.5)
    try:
        for trial in range(1, args.trials + 1):
            master_path = (run_dir / f"trial_{trial:02d}_master.raw").resolve()
            slave_path = (run_dir / f"trial_{trial:02d}_slave.raw").resolve()
            slave.start_recording(str(slave_path))
            master.start_recording(str(master_path))
            time.sleep(args.duration)
            slave.stop_recording()
            master.stop_recording()
            time.sleep(0.2)
            rows.append({"trial": trial, "master_file": str(master_path), "slave_file": str(slave_path)})
            print(f"完成第 {trial}/{args.trials} 段")
    finally:
        slave.stop()
        master.stop()

    for row in rows:
        ms, me = raw_range(Path(row["master_file"]))
        ss, se = raw_range(Path(row["slave_file"]))
        start_delta = ss - ms
        end_delta = se - me
        span_delta = (se - ss) - (me - ms)
        residual = lambda value: value - round(value / args.packet_period_us) * args.packet_period_us
        row.update({
            "master_start_us": ms, "master_end_us": me,
            "slave_start_us": ss, "slave_end_us": se,
            "start_delta_us": start_delta,
            "end_delta_us": end_delta,
            "span_delta_us": span_delta,
            "start_clock_residual_us": residual(start_delta),
            "end_clock_residual_us": residual(end_delta),
            "span_clock_residual_us": residual(span_delta),
        })
    passed = all(
        max(abs(row["start_clock_residual_us"]), abs(row["end_clock_residual_us"]),
            abs(row["span_clock_residual_us"])) <= args.limit_us
        for row in rows
    )
    report = {"master": args.master, "slave": args.slave, "limit_us": args.limit_us,
              "packet_period_us": args.packet_period_us,
              "trials": rows, "passed": passed}
    report_path = run_dir / "endpoint_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"trial {row['trial']}: start {row['start_delta_us']:+d} us, "
            f"end {row['end_delta_us']:+d} us, span {row['span_delta_us']:+d} us; "
            f"clock residual {row['start_clock_residual_us']:+d}/"
            f"{row['end_clock_residual_us']:+d}/{row['span_clock_residual_us']:+d} us"
        )
    print(f"报告: {report_path.resolve()}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
