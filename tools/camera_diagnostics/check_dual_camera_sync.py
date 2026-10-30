"""Configure and measure hardware clock synchronization of two DVSLume cameras.

Point both cameras at the same changing scene. Irregular LED flashes or moving a
high-contrast object in both fields of view gives the most reliable result.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from dvsense_driver.camera_manager import DvsCameraManager
    from dvsense_driver.hal.camera_tool import ToolType
except ImportError as exc:
    raise SystemExit(
        "未安装官方 Python SDK。请执行: python -m pip install dvsense-driver"
    ) from exc


@dataclass
class LagResult:
    lag_us: int
    correlation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配置两台 DVSLume 主从同步，并用事件时间戳验证同步效果。"
    )
    parser.add_argument("--master", help="主相机序列号；省略时使用检测到的第一台相机")
    parser.add_argument("--slave", help="从相机序列号；省略时使用检测到的第二台相机")
    parser.add_argument("--duration", type=float, default=8.0, help="采集秒数，默认 8")
    parser.add_argument(
        "--batch-us", type=int, default=10_000, help="SDK 每批事件的时间长度，默认 10000 us"
    )
    parser.add_argument(
        "--bin-us", type=int, default=1_000, help="互相关统计的时间桶宽度，默认 1000 us"
    )
    parser.add_argument(
        "--max-lag-us", type=int, default=50_000, help="搜索的最大时间偏移，默认 50000 us"
    )
    parser.add_argument(
        "--tolerance-us", type=int, default=2_000, help="判定同步的偏移容差，默认 2000 us"
    )
    parser.add_argument(
        "--min-correlation", type=float, default=0.30, help="可信判定所需的最低相关系数，默认 0.30"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sync_measurement.csv"),
        help="保存事件率时间序列的 CSV 路径",
    )
    parser.add_argument(
        "--no-configure",
        action="store_true",
        help="不修改同步模式，只读取当前模式并测量",
    )
    return parser.parse_args()


def get_sync_tool_type():
    """Handle the missing TOOL_SYNC name in the 1.2.6 Python enum binding."""
    for name in ("SYNC", "TOOL_SYNC"):
        value = getattr(ToolType, name, None)
        if value is not None:
            return value
    return ToolType(2)


def camera_label(desc) -> str:
    return f"{desc.product} / {desc.serial}"


def select_descriptions(descs: Sequence, master_serial: Optional[str], slave_serial: Optional[str]):
    event_cameras = [desc for desc in descs if desc.product != "DVSync"]
    if len(event_cameras) < 2:
        raise RuntimeError(
            f"SDK 只发现 {len(event_cameras)} 台事件相机。请确认两台相机均已连接，并完全退出 "
            "DVSense Insight、Metavision 等可能占用相机的程序。"
        )

    by_serial = {desc.serial: desc for desc in event_cameras}
    if master_serial and master_serial not in by_serial:
        raise RuntimeError(f"找不到主相机序列号 {master_serial}")
    if slave_serial and slave_serial not in by_serial:
        raise RuntimeError(f"找不到从相机序列号 {slave_serial}")

    master = by_serial[master_serial] if master_serial else event_cameras[0]
    slave = by_serial[slave_serial] if slave_serial else next(
        (desc for desc in event_cameras if desc.serial != master.serial), None
    )
    if slave is None:
        raise RuntimeError("主相机和从相机不能是同一台设备")
    if master.serial == slave.serial:
        raise RuntimeError("主相机和从相机不能使用同一个序列号")
    return master, slave


def read_mode(camera, sync_type) -> str:
    ok, mode = camera.get_tool(sync_type).get_param("mode")
    if not ok:
        raise RuntimeError("SDK 无法读取相机同步模式")
    return str(mode)


def set_mode(camera, sync_type, mode: str) -> None:
    if not camera.get_tool(sync_type).set_param("mode", mode):
        raise RuntimeError(f"SDK 无法将相机同步模式设置为 {mode}")
    actual = read_mode(camera, sync_type)
    if actual != mode:
        raise RuntimeError(f"同步模式写入后回读为 {actual}，期望值为 {mode}")


def concatenate_timestamp_batches(batches: Iterable[np.ndarray]) -> np.ndarray:
    timestamps = [batch["timestamp"].astype(np.int64, copy=True) for batch in batches if batch.size]
    if not timestamps:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(timestamps)


def capture_parallel(master, slave, duration_s: float, stall_timeout_s: float = 3.0):
    """Read each camera only from its own thread and fail clearly if either stalls."""
    batches = {"master": [], "slave": []}
    errors = {}
    last_batch_at = {"master": time.perf_counter(), "slave": time.perf_counter()}
    stop_event = threading.Event()
    deadline = time.perf_counter() + duration_s

    def worker(name, camera):
        try:
            while not stop_event.is_set() and time.perf_counter() < deadline:
                batch = camera.get_next_batch()
                batches[name].append(batch)
                last_batch_at[name] = time.perf_counter()
        except Exception as exc:  # The main thread reports both camera states together.
            errors[name] = exc
            stop_event.set()

    threads = {
        "master": threading.Thread(target=worker, args=("master", master), daemon=True),
        "slave": threading.Thread(target=worker, args=("slave", slave), daemon=True),
    }
    for thread in threads.values():
        thread.start()

    stalled = []
    while time.perf_counter() < deadline and not stop_event.wait(0.05):
        now = time.perf_counter()
        stalled = [
            name
            for name, thread in threads.items()
            if thread.is_alive() and now - last_batch_at[name] > stall_timeout_s
        ]
        if stalled:
            stop_event.set()
            break

    stop_event.set()
    for thread in threads.values():
        thread.join(timeout=0.5)

    if errors:
        details = "; ".join(f"{name}: {exc}" for name, exc in errors.items())
        raise RuntimeError(f"相机读取失败: {details}")
    if stalled:
        counts = ", ".join(f"{name}={len(batches[name])} 批" for name in batches)
        raise RuntimeError(
            f"{', '.join(stalled)} 超过 {stall_timeout_s:g} 秒未返回事件批次（{counts}）。"
            "若从机卡住，说明它未收到主机同步时钟；请重点检查 pin 3 SYNC 和 pin 6 GND。"
        )
    return (
        concatenate_timestamp_batches(batches["master"]),
        concatenate_timestamp_batches(batches["slave"]),
    )


def make_rate_traces(
    master_ts: np.ndarray, slave_ts: np.ndarray, bin_us: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(int(master_ts.min()), int(slave_ts.min()))
    end = min(int(master_ts.max()), int(slave_ts.max()))
    if end - start < 20 * bin_us:
        raise RuntimeError("两路时间戳的重叠区间太短，无法计算同步偏移")

    edges = np.arange(start, end + bin_us + 1, bin_us, dtype=np.int64)
    master_rate, _ = np.histogram(master_ts, bins=edges)
    slave_rate, _ = np.histogram(slave_ts, bins=edges)
    centers = edges[:-1] + bin_us // 2
    return centers, np.log1p(master_rate.astype(float)), np.log1p(slave_rate.astype(float))


def correlation_at_lag(master: np.ndarray, slave: np.ndarray, lag_bins: int) -> float:
    # Positive lag means the slave event pattern has a later timestamp.
    if lag_bins >= 0:
        a = master[: len(master) - lag_bins or None]
        b = slave[lag_bins:]
    else:
        a = master[-lag_bins:]
        b = slave[: len(slave) + lag_bins]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def estimate_lag(master: np.ndarray, slave: np.ndarray, bin_us: int, max_lag_us: int) -> LagResult:
    max_bins = max(1, max_lag_us // bin_us)
    candidates = []
    for lag_bins in range(-max_bins, max_bins + 1):
        correlation = correlation_at_lag(master, slave, lag_bins)
        if np.isfinite(correlation):
            candidates.append((correlation, lag_bins))
    if not candidates:
        return LagResult(0, float("nan"))
    correlation, lag_bins = max(candidates)
    return LagResult(lag_bins * bin_us, correlation)


def save_trace(path: Path, centers: np.ndarray, master: np.ndarray, slave: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_us", "master_log_event_count", "slave_log_event_count"])
        writer.writerows(zip(centers, master, slave))


def print_result(
    overall: LagResult,
    first: LagResult,
    second: LagResult,
    tolerance_us: int,
    min_correlation: float,
) -> int:
    print("\n测量结果")
    print(f"  全程偏移（从机 - 主机）: {overall.lag_us:+d} us")
    print(f"  全程相关系数:           {overall.correlation:.3f}")
    print(f"  前半段 / 后半段偏移:    {first.lag_us:+d} / {second.lag_us:+d} us")
    print(f"  观测到的偏移变化:       {second.lag_us - first.lag_us:+d} us")

    correlations = (overall.correlation, first.correlation, second.correlation)
    if not all(np.isfinite(value) for value in correlations) or overall.correlation < min_correlation:
        print("\n结论: 数据不足，不能可靠判断。请让两台相机同时看到更明显、非周期性的亮度变化后重试。")
        return 2
    if abs(overall.lag_us) <= tolerance_us and abs(second.lag_us - first.lag_us) <= tolerance_us:
        print(f"\n结论: 通过。在当前 {tolerance_us} us 容差下，时间戳偏移小且未观察到明显漂移。")
        return 0
    print(f"\n结论: 未通过。偏移或偏移变化超过 {tolerance_us} us，请检查接线、主从角色和启动顺序。")
    return 1


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.batch_us <= 0 or args.bin_us <= 0:
        raise SystemExit("duration、batch-us 和 bin-us 必须大于 0")
    if args.max_lag_us < args.bin_us:
        raise SystemExit("max-lag-us 不能小于 bin-us")

    manager = DvsCameraManager()
    manager.update_cameras()
    descs = manager.get_camera_descs()
    print("SDK 检测到的相机:")
    for desc in descs:
        print(f"  - {camera_label(desc)}")
    master_desc, slave_desc = select_descriptions(descs, args.master, args.slave)
    print(f"\n主相机: {camera_label(master_desc)}")
    print(f"从相机: {camera_label(slave_desc)}")

    master = manager.open_camera(master_desc.serial)
    slave = manager.open_camera(slave_desc.serial)
    sync_type = get_sync_tool_type()
    master_started = False
    slave_started = False
    try:
        if args.no_configure:
            print(f"当前主相机模式: {read_mode(master, sync_type)}")
            print(f"当前从相机模式: {read_mode(slave, sync_type)}")
        else:
            set_mode(master, sync_type, "MASTER")
            set_mode(slave, sync_type, "SLAVE")
            print("同步模式已设置并回读确认: MASTER / SLAVE")

        master.set_batch_events_time(args.batch_us)
        slave.set_batch_events_time(args.batch_us)

        print("正在先启动从相机……")
        slave.start()
        slave_started = True
        time.sleep(0.2)
        print("正在启动主相机；主机应开始输出 1 MHz 同步时钟……")
        master.start()
        master_started = True
        print(f"采集 {args.duration:g} 秒。请持续制造两台相机都能看到的不规则亮度变化。")
        master_ts, slave_ts = capture_parallel(master, slave, args.duration)
    finally:
        if slave_started:
            slave.stop()
        if master_started:
            master.stop()

    print(f"主相机事件数: {len(master_ts):,}; 从相机事件数: {len(slave_ts):,}")
    if len(master_ts) == 0 or len(slave_ts) == 0:
        raise RuntimeError("至少一路没有事件；请增加场景亮度变化后重试")

    master_start, master_end = int(master_ts.min()), int(master_ts.max())
    slave_start, slave_end = int(slave_ts.min()), int(slave_ts.max())
    master_span = master_end - master_start
    slave_span = slave_end - slave_start
    print("时间戳范围（用于核对时钟零点和采集跨度）:")
    print(f"  主相机: {master_start} .. {master_end} us，跨度 {master_span} us")
    print(f"  从相机: {slave_start} .. {slave_end} us，跨度 {slave_span} us")
    print(f"  首事件差（从 - 主）: {slave_start - master_start:+d} us")
    print(f"  末事件差（从 - 主）: {slave_end - master_end:+d} us")
    print(f"  跨度差（从 - 主）:   {slave_span - master_span:+d} us")

    centers, master_rate, slave_rate = make_rate_traces(master_ts, slave_ts, args.bin_us)
    overall = estimate_lag(master_rate, slave_rate, args.bin_us, args.max_lag_us)
    midpoint = len(master_rate) // 2
    first = estimate_lag(master_rate[:midpoint], slave_rate[:midpoint], args.bin_us, args.max_lag_us)
    second = estimate_lag(master_rate[midpoint:], slave_rate[midpoint:], args.bin_us, args.max_lag_us)
    save_trace(args.output, centers, master_rate, slave_rate)
    print(f"测量时间序列已保存到: {args.output.resolve()}")
    return print_result(overall, first, second, args.tolerance_us, args.min_correlation)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中止。")
        sys.exit(130)
    except Exception as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        sys.exit(3)
