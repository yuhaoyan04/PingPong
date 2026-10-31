import numpy as np
import pytest

from pingpong_event_pipeline.event_timing import EVENTS
from pingpong_event_pipeline.raw_sync import IndexedEvents, collect_ranges


def events(times):
    e = np.zeros(len(times), EVENTS); e['timestamp'] = times
    return e


def test_exact_half_open_windows_keep_all_simultaneous_events_and_integer_precision():
    start = 2**53 + 1
    source = events([start-1, start, start, start+1, start+2])
    index = IndexedEvents(source)
    assert index.times.dtype == np.int64 and index.times.flags.c_contiguous
    assert index.window(start, start+2)['timestamp'].tolist() == [start, start, start+1]
    assert not len(index.window(start, start))
    with pytest.raises(ValueError, match='precedes'):
        index.window(start+1, start)


def test_refuse_to_sort_away_a_backwards_timestamp():
    with pytest.raises(ValueError, match='non-monotonic'):
        IndexedEvents(events([100, 103, 102, 104]))
    with pytest.raises(ValueError, match='signed'):
        IndexedEvents(events([2**63]))


def test_sequential_collection_copies_sdk_buffers_and_preserves_boundary_anomalies():
    source = events([1, 3, 3])
    def batches():
        yield source
        source['timestamp'] = [2, 4, 5]
        yield source
    arrays, hist, report = collect_ranges(batches(), [{'serve':'s', 'start_us':2, 'end_us':5}], 0, 6, bin_us=1)
    assert arrays['s']['timestamp'].tolist() == [3, 3, 2, 4]
    assert report['first_backwards'] == {'event_index':3,'before_us':3,'after_us':2}
    assert hist.sum() == 6
