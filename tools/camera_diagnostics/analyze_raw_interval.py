"""Measure correlation lag inside a known common-activity RAW interval."""

import argparse

import numpy as np

from check_dual_camera_sync import estimate_lag


def histogram(path, start_us, end_us, bin_us):
    from dvsense_driver.raw_file_reader import RawFileReader

    reader = RawFileReader(path)
    if not reader.load_file():
        raise RuntimeError(path)
    # The SDK overload takes start timestamp and duration, not end timestamp.
    events = reader.get_n_time_events(start_us, end_us - start_us)
    bins = (end_us - start_us) // bin_us + 1
    result = np.zeros((3, bins), dtype=np.int64)
    if events.size:
        t = (events["timestamp"].astype(np.int64) - start_us) // bin_us
        p = events["polarity"].astype(np.int64) & 1
        valid = (t >= 0) & (t < bins)
        result[0] = np.bincount(t[valid], minlength=bins)
        result[1] = np.bincount(t[valid & (p == 1)], minlength=bins)
        result[2] = np.bincount(t[valid & (p == 0)], minlength=bins)
    return result, int(np.count_nonzero(valid)) if events.size else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master")
    parser.add_argument("slave")
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--max-lag-us", type=int, default=20_000)
    args = parser.parse_args()
    if args.end_us <= args.start_us or args.bin_us <= 0:
        parser.error("end-us must be greater than start-us; bin-us must be positive")
    m, nm = histogram(args.master, args.start_us, args.end_us, args.bin_us)
    s, ns = histogram(args.slave, args.start_us, args.end_us, args.bin_us)
    print(f"区间事件数 主/从: {nm:,}/{ns:,}")
    for index, name in enumerate(("TOTAL", "ON", "OFF")):
        raw = estimate_lag(np.log1p(m[index]), np.log1p(s[index]), args.bin_us, args.max_lag_us)
        # Changes in rate emphasize the leading/trailing edge of the moving object.
        edge = estimate_lag(
            np.diff(np.log1p(m[index])), np.diff(np.log1p(s[index])),
            args.bin_us, args.max_lag_us,
        )
        print(
            f"{name}: rate lag={raw.lag_us:+d} us r={raw.correlation:.3f}; "
            f"edge lag={edge.lag_us:+d} us r={edge.correlation:.3f}"
        )


if __name__ == "__main__":
    main()
