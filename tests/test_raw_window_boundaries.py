from types import SimpleNamespace

import numpy as np

from pingpong_event_pipeline.events import EVENT_DTYPE, RawWindowReader
from pingpong_event_pipeline.events import checked_event_times,corrected_event_times,synchronized_windows,synchronized_batches
import pytest


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


def test_half_open_streaming_windows_retain_next_window_boundary_events():
    def events(t):
        e=np.zeros(len(t),EVENT_DTYPE);e['timestamp']=t;return e
    class Reader:
        info=SimpleNamespace(start_us=1000,end_us=4000)
        def __init__(self):self.batches=iter([events([1000,1999,2000]),events([2500,3000]),events([3999,4000])])
        def seek(self,t):pass
        def next_duration(self,d):return next(self.batches)
    windows=list(synchronized_windows(Reader(),Reader(),1000,1000))
    assert [e['timestamp'].tolist() for _,e,_ in windows]==[[1000,1999],[2000,2500],[3000,3999]]


def test_negative_timestamp_correction_is_signed_and_cannot_wrap_uint64():
    e=np.zeros(3,EVENT_DTYPE);e['timestamp']=[100,100,101]
    assert corrected_event_times(e,-50)['timestamp'].tolist()==[50,50,51]
    assert e['timestamp'].tolist()==[100,100,101]
    with pytest.raises(ValueError,match='underflow'):corrected_event_times(e,-200)
    e['timestamp']=[100,102,101]
    with pytest.raises(ValueError,match='non-monotonic'):checked_event_times(e)


def test_stream_rejects_backwards_jump_between_individually_sorted_batches():
    class Reader:
        info=SimpleNamespace(start_us=1000,end_us=4000)
        def __init__(self):self.values=iter([[1000,1999],[1500,2500]])
        def seek(self,t):pass
        def next_duration(self,d):
            ts=next(self.values);e=np.zeros(len(ts),EVENT_DTYPE);e['timestamp']=ts;return e
    windows=synchronized_windows(Reader(),Reader(),1000,1000)
    next(windows)
    with pytest.raises(ValueError,match='non-monotonic'):next(windows)


def test_disjoint_batch_rejects_sdk_boundary_overshoot():
    class Reader:
        info=SimpleNamespace(start_us=1000,end_us=2000)
        def seek(self,t):pass
        def next_duration(self,d):
            e=np.zeros(2,EVENT_DTYPE);e['timestamp']=[1000,2000];return e
    with pytest.raises(ValueError,match='outside requested'):
        next(synchronized_batches(Reader(),Reader(),1000))
