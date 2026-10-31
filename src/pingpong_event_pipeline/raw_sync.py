"""Reproducible ball-RAW window audit; no per-camera temporal realignment.

Sequential EVT3 events are the rendering source. SDK sequential reads and seeks
are compared separately so a seek artifact cannot silently become sync evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .event_timing import EVENTS, evt3_header, evt3_state, reference_evt3


def sequence_hash(events):
    return hashlib.sha256(np.asarray(events, dtype=EVENTS).tobytes()).hexdigest()


class IndexedEvents:
    """Validate once and avoid copying a strided 13-byte timestamp field per seek."""
    def __init__(self, events):
        self.events = events
        if len(events) and int(events['timestamp'].max()) > np.iinfo(np.int64).max:
            raise ValueError('timestamp exceeds safe signed indexing range')
        # Signed query values against uint64 force NumPy to cast/copy the whole
        # array to float64 per search. Keep a contiguous, exact int64 index.
        self.times = np.ascontiguousarray(events['timestamp'], dtype=np.int64)
        if len(self.times) > 1 and np.any(self.times[1:] < self.times[:-1]):
            raise ValueError('non-monotonic source: refusing to sort away timing anomalies')

    def window(self, start, end):
        if end < start:
            raise ValueError('end precedes start')
        lo, hi = np.searchsorted(self.times, [start, end], side='left')
        return self.events[lo:hi]


def load_serves(dataset):
    dataset = Path(dataset)
    specs = []
    for folder in sorted((dataset / 'unlabeled_inference/F3E').iterdir()):
        if not folder.is_dir():
            continue
        serve = folder.name.rsplit('_', 1)[0]
        spec = {'serve': serve, 'cameras': {}}
        for camera in ['F3E', 'F59']:
            source = dataset / 'unlabeled_inference' / camera / f'{serve}_{camera}'
            meta = json.loads((source / 'metadata.json').read_text())
            with (source / 'manifest.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            spec['cameras'][camera] = {'folder': str(source.resolve()), 'metadata': meta, 'rows': rows}
        a, b = [spec['cameras'][c]['metadata'] for c in ['F3E', 'F59']]
        for key in ['video_first_us', 'video_step_us', 'reference_video']:
            if a[key] != b[key]:
                raise ValueError(f'{serve}: mismatched reference mapping {key}')
        cap = cv2.VideoCapture(str(dataset / a['reference_video']))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        spec.update(reference_first_us=int(a['video_first_us']), reference_step_us=int(a['video_step_us']),
                    reference_frames=count,
                    start_us=int(a['video_first_us']) - 1000,
                    end_us=int(a['video_first_us']) + (count - 1) * int(a['video_step_us']) + 1)
        specs.append(spec)
    return specs


def collect_ranges(batches, specs, activity_origin=None, activity_end=None, bin_us=500):
    pieces = {s['serve']: [] for s in specs}
    previous = None; events_seen = 0; first_backwards = None
    bins = (activity_end - activity_origin + bin_us - 1) // bin_us if activity_origin is not None else 0
    hist = np.zeros((2, bins), np.int64) if bins else None
    outside_histogram_events = 0
    for incoming in batches:
        e = np.asarray(incoming, dtype=EVENTS)
        if not len(e):
            continue
        t = e['timestamp'].astype(np.int64)
        if first_backwards is None:
            bad = np.flatnonzero(np.diff(t) < 0)
            if previous is not None and t[0] < previous:
                first_backwards = {'event_index': events_seen, 'before_us': previous, 'after_us': int(t[0])}
            elif len(bad):
                i = int(bad[0])
                first_backwards = {'event_index': events_seen + i + 1, 'before_us': int(t[i]), 'after_us': int(t[i + 1])}
        events_seen += len(e); previous = int(t[-1])
        low, high = int(t.min()), int(t.max())
        for s in specs:
            if high >= s['start_us'] and low < s['end_us']:
                selected = e[(t >= s['start_us']) & (t < s['end_us'])]
                if len(selected):
                    pieces[s['serve']].append(selected.copy())
        if hist is not None:
            k = (t - activity_origin) // bin_us
            valid = (k >= 0) & (k < bins)
            outside_histogram_events += int(np.count_nonzero(~valid))
            key = (e['polarity'][valid].astype(np.int64) & 1) * bins + k[valid]
            # Avoid allocating an entire recording-sized bincount for every chunk.
            values, counts = np.unique(key, return_counts=True)
            hist.reshape(-1)[values] += counts
    arrays = {name: np.concatenate(parts) if parts else np.empty(0, EVENTS) for name, parts in pieces.items()}
    return arrays, hist, {'events_seen': events_seen, 'first_backwards': first_backwards,
                          'outside_activity_range_events': outside_histogram_events}


def extract_camera(raw, dataset, output, cache, camera):
    from dvsense_driver.raw_file_reader import RawFileReader
    from .events import RawWindowReader
    specs = load_serves(dataset); output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    raw = Path(raw)
    with raw.open('rb') as stream:
        header, start = evt3_header(stream)
    serial = next(line.split()[-1] for line in header if line.startswith('% serial_number '))
    if not serial.endswith(camera):
        raise ValueError('RAW serial does not match camera label')
    state = evt3_state(start)
    origin = start // 1_000_000 * 1_000_000
    activity_end = max(s['end_us'] for s in specs) + 3_000_000
    reference, hist, source_scan = collect_ranges(reference_evt3(raw, state), specs, origin, activity_end)
    for name, e in reference.items():
        np.save(cache / f'{name}_{camera}.npy', e)
    np.savez_compressed(output / f'activity_{camera}.npz', counts=hist, origin_us=origin, bin_us=500)
    print(camera, 'reference extracted', source_scan, flush=True)
    reader = RawFileReader(str(raw))
    if not reader.load_file():
        raise RuntimeError(raw)
    def sdk_batches():
        while True:
            e = reader.get_n_events(200000)
            if e is None or not len(e):
                break
            yield e.copy()
    sdk, _, sdk_scan = collect_ranges(sdk_batches(), specs)
    seek = RawWindowReader(raw)
    results = []
    for s in specs:
        name = s['serve']; e = reference[name]; other = sdk[name]
        chronological = bool(len(e) > 0 and np.all(e['timestamp'][1:] >= e['timestamp'][:-1]))
        record = {'serve': name, 'camera': camera, 'start_us': s['start_us'], 'end_us': s['end_us'],
                  'events': len(e), 'reference_sha256': sequence_hash(e),
                  'sdk_sequential_events': len(other), 'sdk_sequential_sha256': sequence_hash(other),
                  'sdk_sequential_exact_match': bool(np.array_equal(e, other)),
                  'reference_monotonic': chronological}
        requested = seek.window(s['start_us'], s['end_us'])
        record.update(sdk_padded_seek_events=len(requested), sdk_padded_seek_sha256=sequence_hash(requested),
                      sdk_padded_seek_exact_match=bool(np.array_equal(e, requested)))
        if chronological:
            indexed = IndexedEvents(e)
            # Cover EVERY reference and exported crop endpoint, not just a visual sample.
            endpoints = set(s['reference_first_us'] + i * s['reference_step_us'] for i in range(s['reference_frames']))
            endpoints.update(int(row['timestamp_us']) for row in s['cameras'][camera]['rows'])
            for t in sorted(endpoints):
                window = indexed.window(t - 1000, t)
                results.append({'serve': name, 'camera': camera, 'end_us': t, 'start_us': t - 1000,
                                'events': len(window), 'first_event_us': int(window['timestamp'][0]) if len(window) else None,
                                'last_event_us': int(window['timestamp'][-1]) if len(window) else None,
                                'event_sequence_sha256': sequence_hash(window)})
        s['audit'] = record
        print(camera, name, record, flush=True)
    records = [s['audit'] for s in specs]
    report = {'camera': camera, 'raw': str(raw.resolve()), 'serial': serial, 'header': header,
              'source_scan': source_scan, 'sdk_scan': sdk_scan, 'serves': records,
              'all_selected_windows_sdk_reference_agree': all(r['sdk_sequential_exact_match'] for r in records),
              'all_selected_windows_chronological': all(r['reference_monotonic'] for r in records),
              'all_padded_seeks_exact': all(r['sdk_padded_seek_exact_match'] for r in records),
              'two_camera_physical_sync_verified': False,
              'rendering_source': 'sequential independent EVT3; no sorting; absolute half-open windows',
              'reference_parser_flags': {'missing_state': int(state[6]), 'unknown_words': int(state[7]),
                                         'backwards_high': int(state[8]), 'external_triggers': int(state[9])}}
    (output / f'raw_windows_{camera}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from .motion import write_csv
    write_csv(output / f'raw_windows_{camera}.csv', results)
    return report
