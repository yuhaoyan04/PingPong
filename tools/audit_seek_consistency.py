"""Compare chronological SDK reads with time-addressed reads of identical RAW windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dvsense_driver.raw_file_reader import RawFileReader


def load(path):
    reader = RawFileReader(str(path.resolve()))
    if not reader.load_file():
        raise RuntimeError(f'Cannot load {path}')
    ok, start = reader.get_start_timestamp()
    if not ok:
        raise RuntimeError('No file start')
    return reader, int(start)


def canonical(events, start, end):
    ts = events['timestamp'].astype(np.int64, copy=False)
    data = events[(ts >= start) & (ts < end)]
    return data[np.argsort(data['timestamp'], kind='stable')]


def digest(events):
    h = hashlib.sha256()
    for field in ('timestamp', 'x', 'y', 'polarity'):
        h.update(np.ascontiguousarray(events[field]).tobytes())
    return h.hexdigest()


def multiset_difference(a, b):
    """Compare multiplicities, independently of ordering for equal timestamps."""
    key = np.dtype([('timestamp', '<u8'), ('x', '<u2'), ('y', '<u2'), ('polarity', 'u1')])
    def counts(data):
        copy = np.empty(len(data), dtype=key)
        for name in key.names:
            copy[name] = data[name]
        return np.unique(copy, return_counts=True)
    av, ac = counts(a); bv, bc = counts(b)
    joined = np.union1d(av, bv)
    left = np.zeros(len(joined), np.int32); right = np.zeros(len(joined), np.int32)
    left[np.searchsorted(joined, av)] = ac
    right[np.searchsorted(joined, bv)] = bc
    missing = joined[left != right]
    return {'multiset_equal': not len(missing), 'different_unique_events': int(len(missing)),
            'sequential_excess': int(np.maximum(left-right, 0).sum()),
            'indexed_excess': int(np.maximum(right-left, 0).sum()),
            'different_timestamp_min_us': int(missing['timestamp'].min()) if len(missing) else None,
            'different_timestamp_max_us': int(missing['timestamp'].max()) if len(missing) else None}


def inspect(path, offsets, window_us, chunk_us=50000, padding_us=0):
    sequential, start = load(path)
    windows = [(start+round(v*1e6), start+round(v*1e6)+window_us) for v in offsets]
    pieces = [[] for _ in windows]
    while not sequential.reached_end_of_events():
        before = sequential.get_current_pos_timestamp()
        events = sequential.get_n_time_events(chunk_us)
        after = sequential.get_current_pos_timestamp()
        if len(events):
            ts = events['timestamp'].astype(np.int64, copy=False)
            for i, (lo, hi) in enumerate(windows):
                if ts[-1] >= lo and ts[0] < hi:
                    selected = events[(ts >= lo) & (ts < hi)]
                    if len(selected): pieces[i].append(selected.copy())
        if after >= max(hi for _, hi in windows) or sequential.reached_end_of_events(): break
        if after <= before: raise RuntimeError(f'sequential reader stalled at {after}')
    random_reader, _ = load(path)
    results = []
    for (lo, hi), blocks in zip(windows, pieces):
        if not blocks:
            raise RuntimeError(f'No sequential events in {lo}..{hi}')
        serial = canonical(np.concatenate(blocks), lo, hi)
        indexed = canonical(random_reader.get_n_time_events(lo-padding_us, hi-lo+padding_us), lo, hi)
        shash, ihash = digest(serial), digest(indexed)
        detail = multiset_difference(serial, indexed) if shash != ihash else {'multiset_equal': True}
        results.append({'relative_start_s': (lo-start)/1e6, 'duration_us': hi-lo,
                        'sequential_count': len(serial), 'indexed_count': len(indexed),
                        'sequential_sha256': shash, 'indexed_sha256': ihash,
                        'same_event_sequence': shash == ihash, 'multiset_comparison': detail})
    return {'file': str(path.resolve()), 'start_us': start, 'sequential_chunk_us': chunk_us,
            'indexed_start_padding_us': padding_us, 'windows': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--offset-s', nargs='+', type=float, required=True)
    p.add_argument('--window-us', type=int, default=20000)
    p.add_argument('--chunk-us', type=int, default=50000)
    p.add_argument('--padding-us', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if any(v * 1e6 < args.padding_us for v in args.offset_s) or args.window_us <= 0 or args.chunk_us <= 0 or args.padding_us < 0:
        p.error('invalid window')
    reports = []
    for path in args.files:
        print('COMPARING', path, flush=True)
        reports.append(inspect(path, args.offset_s, args.window_us, args.chunk_us, args.padding_us))
        print(json.dumps(reports[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
