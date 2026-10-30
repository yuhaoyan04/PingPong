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
