"""Exact integer decoded-event deltas and independent EVT3 byte-stream checks.

1 us timestamp quantization is NOT a calibrated 1 us physical accuracy claim.
No sorting, clipping, deduplication, or replacement with batch/host timing.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit

EVENTS = np.dtype([("x", "<u2"), ("y", "<u2"), ("polarity", "u1"), ("timestamp", "<u8")])


def stream_sha256(path):
    """Bounded-memory hashing, including Python 3.9/3.10 SDK environments."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class AdjacentIntervals:
    def __init__(self):
        self.events = 0
        self.previous = None
        self.first = None
        self.histogram = Counter()
        self.negative_examples = []
        self.sequence_hash = hashlib.sha256()

    def ingest(self, events):
        e = np.array(events, dtype=EVENTS, copy=True)
        if not len(e):
            return e, np.empty(0, np.int64), np.empty(0, bool)
        if int(e["timestamp"].max()) > np.iinfo(np.int64).max:
            raise ValueError("timestamp exceeds safe signed subtraction range")
        t = e["timestamp"].astype(np.int64)
        dt = np.zeros(len(e), np.int64)
        valid = np.ones(len(e), bool)
        if self.previous is None:
            valid[0] = False
            self.first = int(t[0])
        else:
            dt[0] = int(t[0]) - self.previous
        dt[1:] = np.diff(t)
        values, counts = np.unique(dt[valid], return_counts=True)
        self.histogram.update({int(v): int(n) for v, n in zip(values, counts)})
        for i in np.flatnonzero(valid & (dt < 0))[:max(0, 20 - len(self.negative_examples))]:
            self.negative_examples.append({"event_index": self.events + int(i),
                                           "timestamp_us": int(t[i]), "delta_us": int(dt[i]),
                                           "crosses_sdk_batch_boundary": bool(i == 0)})
        self.events += len(e)
        self.previous = int(t[-1])
        self.sequence_hash.update(e.tobytes())
        return e, dt, valid

    def summary(self):
        total = sum(self.histogram.values())
        keys = sorted(self.histogram)
        positive = [k for k in keys if k > 0]
        def quantile(q):
            if not total:
                return None
            target = int(np.floor(q * (total - 1)))
            count = 0
            for k in keys:
                count += self.histogram[k]
                if count > target:
                    return k
        return {"events": self.events, "adjacent_intervals": total,
                "first_event_us": self.first, "last_event_us": self.previous,
                "negative_intervals": sum(v for k, v in self.histogram.items() if k < 0),
                "same_timestamp_intervals": self.histogram.get(0, 0),
                "positive_intervals": sum(self.histogram[k] for k in positive),
                "minimum_positive_delta_us": min(positive) if positive else None,
                "observed_positive_delta_gcd_us": int(np.gcd.reduce(positive)) if positive else None,
                "minimum_delta_us": keys[0] if keys else None, "maximum_delta_us": keys[-1] if keys else None,
                "delta_quantiles_us": {str(q): quantile(q) for q in [.01, .5, .95, .99]},
                "zero_fraction": self.histogram.get(0, 0) / total if total else None,
                "negative_examples": self.negative_examples,
                "event_sequence_sha256": self.sequence_hash.hexdigest()}


def evt3_header(stream):
    lines = []
    for _ in range(1000):
        line = stream.readline()
        if not line or not line.startswith(b"%"):
            raise ValueError("missing EVT3 RAW text header/end marker")
        decoded = line.decode("ascii", errors="strict").strip()
        lines.append(decoded)
        if decoded == "% end":
            break
    else:
        raise ValueError("oversized RAW header")
    if not any("EVT3" in line or line == "% evt 3.0" for line in lines):
        raise ValueError("reference decoder only supports EVT3")
    start = next((int(line.split()[-1]) for line in lines if line.startswith("% dvs_start_timestamp ")), 0)
    return lines, start


def evt3_state(start_timestamp_us=0):
    state = np.zeros(13, np.int64)
    state[0] = -1  # last time-high
    state[2] = int(start_timestamp_us) // (1 << 24) * (1 << 24)
    state[3:5] = -1  # last Y, vector base X
    state[12] = int(start_timestamp_us)
    return state


@njit(cache=True)
def decode_evt3_words(words, state):
    """Reference implementation from the published EVT3 bit layout.

    State persists through arbitrary byte-chunk boundaries and 24-bit wraps.
    Timestamp updates/markers are not CD events. Vector masks emit individual
    same-timestamp pixel events in ascending bit order.
    """
    out = np.empty(len(words) * 12, dtype=EVENTS)
    count = 0
    for word in words:
        kind, value = int(word) >> 12, int(word) & 4095
        if kind == 8:
            if state[0] < 0:
                reference_low = state[12] % (1 << 24)
                if (value << 12) < reference_low - (1 << 23):
                    state[2] += 1 << 24
            elif value < state[0]:
                if state[0] - value > 2048:
                    state[2] += 1 << 24
                else:
                    state[8] += 1  # malformed/non-wrap backwards high
            if value != state[0]:
                state[1] = 0
            state[0] = value
        elif kind == 6:
            state[1] = value
        elif kind == 0:
            state[3] = value & 2047
            state[11] |= 1 << ((value >> 11) & 1)
        elif kind == 3:
            state[4], state[5] = value & 2047, (value >> 11) & 1
        elif kind == 2 or kind == 4 or kind == 5:
            nbits = 1 if kind == 2 else (12 if kind == 4 else 8)
            for bit in range(nbits):
                if kind != 2 and not ((value >> bit) & 1):
                    continue
                if state[0] < 0 or state[3] < 0 or (kind != 2 and state[4] < 0):
                    state[6] += 1
                    continue
                out[count]["x"] = (value & 2047) if kind == 2 else state[4] + bit
                out[count]["y"] = state[3]
                out[count]["polarity"] = ((value >> 11) & 1) if kind == 2 else state[5]
                out[count]["timestamp"] = state[2] + (state[0] << 12) + state[1]
                count += 1
            if kind != 2:
                state[4] += nbits
        elif kind == 10:
            state[9] += 1
        elif kind != 7 and kind != 14 and kind != 15:
            state[7] += 1
    return out[:count].copy()


def reference_evt3(path, state, word_batch=65536):
    with Path(path).open("rb") as stream:
        evt3_header(stream)
        while True:
            block = stream.read(word_batch * 2)
            if not block:
                break
            if len(block) % 2:
                raise ValueError("truncated 16-bit EVT3 word at file end")
            yield decode_evt3_words(np.frombuffer(block, dtype="<u2"), state)


@njit(cache=True)
def _count_event_clock(events, first_us, counts):
    outside = 0
    for event in events:
        i = int(event['timestamp']) - first_us
        if i < 0 or i >= len(counts):
            outside += 1
        else:
            counts[i] += 1
    return outside


def export_clock_runs(path, timing_report, output):
    """Lossless ordered event times without duplicating every pixel record.

    counts[k] consecutive events have timestamp first_us+k. For verified
    monotonic input this represents every original adjacent delta, including
    zeros and batch boundaries. The very first event has no previous delta.
    """
    path, output = Path(path), Path(output)
    report = json.loads(Path(timing_report).read_text())
    if not report['decoded_intervals_verified'] or Path(report['raw']).resolve() != path.resolve():
        raise ValueError('requires a verified timing report for this RAW')
    reference = report['reference_evt3']
    first, last = reference['first_event_us'], reference['last_event_us']
    length = last-first+1
    if length <= 0 or length > 100_000_000:
        raise ValueError('clock run exporter supports at most 100 seconds per recording')
    counts = np.zeros(length, np.uint32)
    with path.open('rb') as stream:
        _, start = evt3_header(stream)
    state = evt3_state(start); sequence = hashlib.sha256(); total = 0; outside = 0
    for e in reference_evt3(path, state):
        outside += _count_event_clock(e, first, counts)
        sequence.update(e.tobytes()); total += len(e)
    if outside or total != reference['events'] or sequence.hexdigest() != reference['event_sequence_sha256']:
        raise ValueError('current event stream differs from the verified timing report')
    if int(counts.max()) <= np.iinfo(np.uint16).max:
        counts = counts.astype(np.uint16)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, counts=counts, first_timestamp_us=np.uint64(first),
                        last_timestamp_us=np.uint64(last), events=np.uint64(total),
                        event_sequence_sha256=np.array(sequence.hexdigest()), timestamp_quantum_us=np.uint64(1))
    return {'file': str(output.resolve()), 'events': total, 'timestamp_bins': len(counts),
            'maximum_events_per_timestamp': int(counts.max()), 'source_event_sequence_sha256': sequence.hexdigest(),
            'encoding': 'counts[k] consecutive events at first_timestamp_us+k; first event has no previous delta',
            'size_bytes': output.stat().st_size}


def audit_raw(path, output, batch_events=200000, export_events=True):
    from dvsense_driver.raw_file_reader import RawFileReader
    path, output = Path(path), Path(output)
    if batch_events <= 0:
        raise ValueError("batch_events must be positive")
    output.mkdir(parents=True, exist_ok=False)
    with path.open("rb") as stream:
        header, raw_start = evt3_header(stream)
    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"cannot load {path}")
    _, file_start = reader.get_start_timestamp()
    _, file_end = reader.get_end_timestamp()
    acc = AdjacentIntervals()
    sample_path = output / "adjacent_events_sample.csv"
    termination, invalid_coordinates = None, 0
    previous_xypt = None
    sample_count = 0
    try:
        with sample_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event_index", "previous_x", "previous_y", "previous_polarity", "previous_timestamp_us", "x", "y", "polarity", "timestamp_us", "delta_us", "delta_s", "crosses_sdk_batch_boundary"])
            writer.writeheader()
            part = 0
            while True:
                incoming = reader.get_n_events(batch_events)
                if incoming is None or not len(incoming):
                    termination = "empty_read"
                    break
                first_index = acc.events
                e, dt, valid = acc.ingest(incoming)  # copy BEFORE the next SDK call
                invalid_coordinates += int(np.count_nonzero((e["x"] >= reader.get_width()) | (e["y"] >= reader.get_height())))
                if export_events:
                    np.savez_compressed(output / f"events_{part:05d}.npz", events=e, delta_us=dt,
                                        delta_valid=valid, first_global_event_index=np.int64(first_index))
                sample_indices = set(range(max(0, min(len(e), 5000 - sample_count))))
                if first_index:
                    sample_indices.add(0)  # explicitly expose every cross-batch delta
                for i in sorted(sample_indices):
                    previous = previous_xypt if i == 0 else e[i - 1]
                    row = {"event_index": first_index + i, "x": int(e[i]["x"]), "y": int(e[i]["y"]),
                           "polarity": int(e[i]["polarity"]), "timestamp_us": int(e[i]["timestamp"]),
                           "delta_us": int(dt[i]) if valid[i] else "", "delta_s": float(dt[i]) * 1e-6 if valid[i] else "",
                           "crosses_sdk_batch_boundary": bool(i == 0 and first_index > 0)}
                    for key in ["x", "y", "polarity", "timestamp"]:
                        row["previous_" + key + ("_us" if key == "timestamp" else "")] = int(previous[key]) if previous is not None else ""
                    writer.writerow(row)
                    sample_count += 1
                previous_xypt = e[-1].copy()
                part += 1
                # The SDK end timestamp/EOF flag can precede a remaining partial
                # event batch. Only an empty read establishes stream exhaustion;
                # the independent byte decoder then checks completeness.
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    ref = AdjacentIntervals()
    state = evt3_state(raw_start)
    for e in reference_evt3(path, state):
        ref.ingest(e)
    sdk, reference = acc.summary(), ref.summary()
    agreement = sdk["events"] == reference["events"] and sdk["event_sequence_sha256"] == reference["event_sequence_sha256"]
    history = None
    historical_path = path.parent / "monotonicity_report.json"
    if historical_path.is_file():
        records = json.loads(historical_path.read_text(encoding="utf-8"))
        if isinstance(records, list):
            history = next((r for r in records if r["file"].replace("\\", "/").split("/")[-1] == path.name), None)
    matches_history = None if history is None else (sdk["events"] == history["events"] and sdk["last_event_us"] == history["last_event_us"])
    # Ball recordings are multi-GB: hash without retaining the RAW in RAM.
    raw_sha256 = stream_sha256(path)
    report = {"raw": str(path.resolve()), "raw_size_bytes": path.stat().st_size,
              "raw_sha256": raw_sha256, "header": header,
              "timestamp_unit": "microseconds", "timestamp_quantum_us": 1,
              "formula": "delta_us[i] = int64(timestamp[i]) - int64(timestamp[i-1])",
              "sdk_batch_events": batch_events, "sdk_termination": termination,
              "sdk_file_start_us": int(file_start), "sdk_file_end_us": int(file_end),
              "sdk": sdk, "reference_evt3": reference, "sdk_reference_exact_agreement": agreement,
              "reference_missing_state_events": int(state[6]), "reference_unknown_words": int(state[7]),
              "reference_backwards_time_high": int(state[8]), "external_trigger_words": int(state[9]),
              "invalid_coordinates": invalid_coordinates, "historical_record": history,
              "current_file_matches_historical_record": matches_history,
              "decoded_intervals_verified": bool(agreement and sdk["events"] > 1 and sdk["negative_intervals"] == 0
                                                and not invalid_coordinates and not np.any(state[6:9])),
              "physical_time_accuracy_calibrated": False, "training_recording_timing_verified": False,
              "two_camera_synchronization_verified": False,
              "interpretation": ["Zero delta means the same 1 us bin, not proven physical simultaneity.",
                                 "Batch size/USB timing/crop frame interval are not event delta.",
                                 "Original order is preserved; negative deltas are retained and reported.",
                                 "Byte-level agreement verifies the current file, not absent tail data or clock accuracy.",
                                 "An individual RAW audit alone does not establish two-camera synchronization or its mapping to exported training frames."],
              "exports": {"full_events": "events_*.npz" if export_events else None,
                          "sample": "adjacent_events_sample.csv", "histogram": "delta_histogram.csv"}}
    with (output / "delta_histogram.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["delta_us", "count"])
        writer.writerows(sorted(acc.histogram.items()))
    (output / "timing_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
