from types import SimpleNamespace

import numpy as np

from pingpong_event_pipeline.events import EVENT_DTYPE, RawWindowReader


def test_window_pads_seek_and_returns_half_open_interval():
    events = np.zeros(5, dtype=EVENT_DTYPE)
    events['timestamp'] = [1200, 1999, 2000, 2999, 3000]
    calls = []
    class Reader:
        def get_n_time_events(self, start, duration):
            calls.append((start, duration))
            return events
    raw = RawWindowReader.__new__(RawWindowReader)
    raw.reader = Reader()
    raw.info = SimpleNamespace(start_us=1000)
    result = raw.window(2000, 3000)
    assert calls == [(1000, 2000)]
    assert result['timestamp'].tolist() == [2000, 2999]
    assert not len(raw.window(3000, 3000))


def test_duration_batch_does_not_alias_the_sdk_reusable_buffer():
    buffer = np.zeros(2, dtype=EVENT_DTYPE)
    class Reader:
        count = 0
        def get_n_time_events(self, duration):
            self.count += 1
            buffer['timestamp'] = [self.count * 100, self.count * 100 + 1]
            return buffer
    raw = RawWindowReader.__new__(RawWindowReader)
    raw.reader = Reader()
    first = raw.next_duration(100)
    second = raw.next_duration(100)
    assert first['timestamp'].tolist() == [100, 101]
    assert second['timestamp'].tolist() == [200, 201]
    assert not np.shares_memory(first, buffer)
