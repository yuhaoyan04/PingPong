import numpy as np
import pytest

from pingpong_event_pipeline.event_timing import AdjacentIntervals, EVENTS, decode_evt3_words, evt3_state


def events(times):
    out = np.zeros(len(times), dtype=EVENTS)
    out["timestamp"] = times
    return out


def test_exact_signed_deltas_include_zero_negative_and_batch_boundary():
    acc = AdjacentIntervals()
    _, d1, valid = acc.ingest(events([100, 100, 103]))
    assert d1.tolist() == [0, 0, 3] and valid.tolist() == [False, True, True]
    _, d2, valid = acc.ingest(events([105, 104, 107]))
    assert d2.tolist() == [2, -1, 3] and valid.all()
    report = acc.summary()
    assert report["events"] == 6 and report["adjacent_intervals"] == 5
    assert report["negative_intervals"] == 1 and report["same_timestamp_intervals"] == 1
    assert report["negative_examples"][0]["delta_us"] == -1


def test_reference_evt3_vectors_and_repeated_time_high_do_not_create_events():
    words = np.array([0x8001, 0x6005, 0x0003, 0x2807, 0x8001, 0x2008,
                      0x380a, 0x4005, 0x5003], dtype=np.uint16)
    decoded = decode_evt3_words(words, evt3_state())
    assert decoded["timestamp"].tolist() == [4101] * 6
    assert decoded["x"].tolist() == [7, 8, 10, 12, 22, 23]
    assert decoded["y"].tolist() == [3] * 6
    assert decoded["polarity"].tolist() == [1, 0, 1, 1, 1, 1]


def test_evt3_wrap_and_arbitrary_chunk_boundaries():
    words = np.array([0x8fff, 0x6ffe, 0x0002, 0x2003, 0x8000, 0x6001, 0x2004], dtype=np.uint16)
    whole = decode_evt3_words(words, evt3_state())
    state = evt3_state()
    parts = [decode_evt3_words(words[:3], state), decode_evt3_words(words[3:5], state), decode_evt3_words(words[5:], state)]
    combined = np.concatenate(parts)
    np.testing.assert_array_equal(combined, whole)
    assert whole["timestamp"].tolist() == [(1 << 24) - 2, (1 << 24) + 1]
    assert np.diff(whole["timestamp"].astype(np.int64)).tolist() == [3]


def test_sequence_hash_does_not_depend_on_sdk_batch_size():
    values = events([1, 1, 2, 100, 101])
    a, b = AdjacentIntervals(), AdjacentIntervals()
    a.ingest(values)
    b.ingest(values[:2]); b.ingest(values[2:])
    assert a.summary() == b.summary()


def test_first_event_has_no_fabricated_previous_interval():
    acc = AdjacentIntervals()
    acc.ingest(events([100]))
    assert acc.summary()["adjacent_intervals"] == 0


def test_uint64_subtraction_cannot_underflow_into_huge_positive_interval():
    acc = AdjacentIntervals(); _, dt, _ = acc.ingest(events([100, 90]))
    assert dt.tolist() == [0, -10]
    with pytest.raises(ValueError, match="signed"):
        AdjacentIntervals().ingest(events([2**63 + 1]))


def test_raw_audit_drains_partial_batch_despite_premature_sdk_eof(tmp_path, monkeypatch):
    import sys
    import types
    from pingpong_event_pipeline.event_timing import audit_raw
    words = np.array([0x8001, 0x6005, 0x0002, 0x2003, 0x2004, 0x2005], np.uint16)
    source = decode_evt3_words(words, evt3_state())
    path = tmp_path / 'ball.raw'
    path.write_bytes(b'% evt 3.0\n% dvs_start_timestamp 0\n% end\n' + words.astype('<u2').tobytes())
    class Reader:
        def __init__(self, path): self.parts = iter([source[:2], source[2:], source[:0]])
        def load_file(self): return True
        def get_start_timestamp(self): return True, 4101
        def get_end_timestamp(self): return True, 4101
        def get_width(self): return 1280
        def get_height(self): return 720
        def get_n_events(self, count): return next(self.parts)
        def reached_end_of_events(self): return True  # tail batch still remains!
    monkeypatch.setitem(sys.modules, 'dvsense_driver.raw_file_reader', types.SimpleNamespace(RawFileReader=Reader))
    report = audit_raw(path, tmp_path / 'out', batch_events=2, export_events=False)
    assert report['sdk']['events'] == 3
    assert report['sdk_reference_exact_agreement']
    assert report['decoded_intervals_verified']
    assert report['sdk_termination'] == 'empty_read'


def test_lossless_clock_runs_recover_every_adjacent_delta(tmp_path):
    import json
    from pingpong_event_pipeline.event_timing import export_clock_runs
    words = np.array([0x8001,0x6005,0x0002,0x2003,0x2004,0x2005,0x6006,0x2006],np.uint16)
    source = decode_evt3_words(words,evt3_state())
    path = tmp_path/'events.raw'
    path.write_bytes(b'% evt 3.0\n% dvs_start_timestamp 0\n% end\n'+words.astype('<u2').tobytes())
    acc=AdjacentIntervals(); acc.ingest(source)
    report={'raw':str(path),'decoded_intervals_verified':True,'reference_evt3':acc.summary()}
    report_path=tmp_path/'timing.json'; report_path.write_text(json.dumps(report))
    out=tmp_path/'clock.npz'; result=export_clock_runs(path,report_path,out)
    encoded=np.load(out)
    recovered=np.repeat(np.arange(len(encoded['counts']),dtype=np.int64)+int(encoded['first_timestamp_us']),encoded['counts'])
    assert result['events']==4
    assert recovered.tolist()==[4101,4101,4101,4102]
    assert np.diff(recovered).tolist()==[0,0,1]
    report['decoded_intervals_verified']=False;report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='verified'):
        export_clock_runs(path,report_path,tmp_path/'unverified.npz')
