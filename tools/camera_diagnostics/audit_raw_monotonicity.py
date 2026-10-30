"""Inspect decoded event timestamps in RAW recordings, including batch boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np
from dvsense_driver.raw_file_reader import RawFileReader


def inspect(path, chunk_us):
    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"Cannot load {path}")
    ok_start, start = reader.get_start_timestamp()
    ok_end, end = reader.get_end_timestamp()
    if not ok_start or not ok_end or end <= start:
        raise RuntimeError(f"Invalid RAW time range: {path}")
    count = backward = empty = 0
    first = previous = None
    min_step = None
    backward_examples = []
    batches = 0
    while not reader.reached_end_of_events():
        before = reader.get_current_pos_timestamp()
        events = reader.get_n_time_events(chunk_us)
        after = reader.get_current_pos_timestamp()
        batches += 1
        if events.size:
            ts = events['timestamp'].astype(np.int64, copy=False)
            if first is None:
                first = int(ts[0])
            steps = np.diff(ts)
            if previous is not None:
                steps = np.concatenate((np.array([int(ts[0]) - previous]), steps))
            if steps.size:
                backward += int(np.count_nonzero(steps < 0))
                if len(backward_examples) < 20:
                    paired = np.concatenate((np.array([previous], dtype=np.int64), ts)) if previous is not None else ts
                    for index in np.flatnonzero(steps < 0)[:20 - len(backward_examples)]:
                        backward_examples.append({'batch': batches, 'before_us': int(paired[index]),
                                                  'after_us': int(paired[index + 1]),
                                                  'delta_us': int(steps[index]),
                                                  'at_batch_boundary': bool(previous is not None and index == 0)})
                step = int(steps.min())
                min_step = step if min_step is None else min(min_step, step)
            previous = int(ts[-1])
            count += int(ts.size)
        else:
            empty += 1
        if after >= end or reader.reached_end_of_events():
            break
        if after <= before:
            raise RuntimeError(f"Reader did not advance at {after} in {path}")
    close = getattr(reader, 'close', None)
    if close:
        close()
    return dict(file=str(path.resolve()), file_start_us=int(start), file_end_us=int(end),
                events=count, first_event_us=first, last_event_us=previous, batches=batches,
                empty_batches=empty, backward_steps=backward, minimum_step_us=min_step,
                backward_examples=backward_examples,
                passed=bool(count and backward == 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--chunk-us', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_us <= 0:
        parser.error('--chunk-us must be positive')
    reports = [inspect(path, args.chunk_us) for path in args.files]
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if all(report['passed'] for report in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
