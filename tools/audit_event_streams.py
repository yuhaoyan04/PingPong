"""Summarize RAW event rates, polarity and timestamp order without changing inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from dvsense_driver.raw_file_reader import RawFileReader


def inspect(path, bin_us=10000, chunk_us=20000):
    reader = RawFileReader(str(path.resolve()))
    if not reader.load_file():
        raise RuntimeError(f'Cannot load {path}')
    ok_start, start = reader.get_start_timestamp()
    ok_end, end = reader.get_end_timestamp()
    if not ok_start or not ok_end or end <= start:
        raise RuntimeError(f'Invalid RAW range: {path}')
    start, end = int(start), int(end)
    bins = np.zeros(((end - start) // bin_us + 2, 2), dtype=np.int64)
    first = last = None
    back = outside = total = 0
    min_step = None
    while not reader.reached_end_of_events():
        before = reader.get_current_pos_timestamp()
        events = reader.get_n_time_events(chunk_us)
        after = reader.get_current_pos_timestamp()
        if events.size:
            times = events['timestamp'].astype(np.int64, copy=False)
            if first is None:
                first = int(times[0])
            steps = np.diff(times)
            if last is not None:
                steps = np.concatenate(([int(times[0]) - last], steps))
            if steps.size:
                back += int(np.count_nonzero(steps < 0))
                v = int(steps.min())
                min_step = v if min_step is None else min(min_step, v)
            last = int(times[-1]); total += len(times)
            valid = (times >= start) & (times <= end)
            outside += int(np.count_nonzero(~valid))
            indices = ((times[valid] - start) // bin_us).astype(np.int64)
            polarities = events['polarity'][valid].astype(np.int64) & 1
            bins += np.bincount(indices * 2 + polarities, minlength=bins.size).reshape(bins.shape)
        if after >= end or reader.reached_end_of_events():
            break
        if after <= before:
            raise RuntimeError(f'Reader stalled at {after} in {path}')
    sums = bins.sum(axis=1)
    duration = (end - start) / 1e6
    return {'file': str(path.resolve()), 'file_start_us': start, 'file_end_us': end,
            'duration_s': duration, 'decoded_events': int(total), 'outside_header_range': outside,
            'first_event_us': first, 'last_event_us': last, 'backward_steps': back,
            'minimum_step_us': min_step, 'polarity_counts': {'off': int(bins[:, 0].sum()), 'on': int(bins[:, 1].sum())},
            'average_mev_s': float(total / duration / 1e6), 'bin_us': bin_us,
            'bin_event_count_percentiles': {str(p): float(np.percentile(sums, p)) for p in (0, 50, 90, 95, 99, 100)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    records = []
    for file in args.files:
        print(f'Analysing {file}', flush=True)
        records.append(inspect(file))
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
