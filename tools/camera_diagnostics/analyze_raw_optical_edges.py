"""Find common optical edges in two already-recorded DVSLume RAW files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_polarity_histogram(path: Path, bin_us: int):
    from dvsense_driver.raw_file_reader import RawFileReader

    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"不能读取 {path}")
    ok, end_us = reader.get_end_timestamp()
    if not ok:
        raise RuntimeError(f"不能读取 {path} 的结束时间")
    bins = int(end_us // bin_us) + 2
    hist = np.zeros((2, bins), dtype=np.int64)
    while not reader.reached_end_of_events():
        events = reader.get_n_time_events(20_000)
        if events.size:
            t = events["timestamp"].astype(np.int64, copy=False) // bin_us
            p = events["polarity"].astype(np.int64, copy=False) & 1
            valid = t < bins
            key = p[valid] * bins + t[valid]
            hist += np.bincount(key, minlength=2 * bins).reshape(2, bins)
        if reader.get_current_pos_timestamp() + 20_000 >= end_us:
            break
    close = getattr(reader, "close", None)
    if close:
        close()
    return hist


def robust_z(x):
    x = x.astype(float)
    median = np.median(x)
    scale = np.median(np.abs(x - median)) * 1.4826
    if scale < 1e-9:
        scale = np.std(x) + 1e-9
    return (x - median) / scale


def corr_at_lag(a, b, lag):
    if lag < 0:
        a, b = a[-lag:], b[:lag]
    elif lag > 0:
        a, b = a[:-lag], b[lag:]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def best_lag(a, b, max_lag):
    candidates = [(corr_at_lag(a, b, lag), lag) for lag in range(-max_lag, max_lag + 1)]
    return max(candidates, key=lambda item: -np.inf if not np.isfinite(item[0]) else item[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("slave", type=Path)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--max-lag-us", type=int, default=5_000)
    args = parser.parse_args()

    print("读取主机极性直方图……")
    m = read_polarity_histogram(args.master, args.bin_us)
    print("读取从机极性直方图……")
    s = read_polarity_histogram(args.slave, args.bin_us)
    n = min(m.shape[1], s.shape[1])
    # Ignore start/stop transients and compare ON, OFF and polarity imbalance.
    # Camera start/stop creates strong shared transients unrelated to the user's
    # optical stimulus. Exclude a full second at both ends.
    margin = max(1, 1_000_000 // args.bin_us)
    features_m = [m.sum(axis=0), m[1], m[0], m[1] - m[0]]
    features_s = [s.sum(axis=0), s[1], s[0], s[1] - s[0]]
    names = ["TOTAL", "ON", "OFF", "ON-OFF"]
    max_lag = args.max_lag_us // args.bin_us
    best = None
    for name, fm, fs in zip(names, features_m, features_s):
        a = robust_z(fm[margin : n - margin])
        b = robust_z(fs[margin : n - margin])
        # Keep optical edges and suppress the ERC-controlled baseline.
        threshold_a = np.quantile(np.abs(a), 0.95)
        threshold_b = np.quantile(np.abs(b), 0.95)
        a = np.where(np.abs(a) >= threshold_a, a, 0.0)
        b = np.where(np.abs(b) >= threshold_b, b, 0.0)
        corr, lag = best_lag(a, b, max_lag)
        candidate = (corr, name, lag * args.bin_us)
        if best is None or (np.isfinite(corr) and corr > best[0]):
            best = candidate
        top_m = np.argsort(np.abs(a))[-30:]
        top_s = np.argsort(np.abs(b))[-30:]
        nearest = []
        for tm in top_m:
            delta = int(top_s[np.argmin(np.abs(top_s - tm))] - tm) * args.bin_us
            if abs(delta) <= args.max_lag_us:
                nearest.append(delta)
        exact_bins = sorted(set(map(int, top_m)) & set(map(int, top_s)))
        exact = len(exact_bins)
        mid = len(a) // 2
        corr_first, lag_first = best_lag(a[:mid], b[:mid], max_lag)
        corr_second, lag_second = best_lag(a[mid:], b[mid:], max_lag)
        print(f"{name}: 匹配强边沿偏移样本 {sorted(nearest)}")
        print(
            f"  最强30个时间窗精确重合 {exact}/30；"
            f"前/后半段峰值 {lag_first * args.bin_us:+d}/{lag_second * args.bin_us:+d} us "
            f"(r={corr_first:.3f}/{corr_second:.3f})"
        )
        if exact_bins:
            exact_ms = [round((index + margin) * args.bin_us / 1000, 1) for index in exact_bins]
            print(f"  精确重合时间（ms）: {exact_ms}")
    print(f"最佳极性证据: {best[1]}, 从-主偏移 {best[2]:+d} us, 相关系数 {best[0]:.3f}")


if __name__ == "__main__":
    main()
