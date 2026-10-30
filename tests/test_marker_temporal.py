import pytest

from pingpong_event_pipeline.marker_temporal import MarkerTracker


def test_gap_is_prediction_then_expires():
    tracker = MarkerTracker(gate_px=20, max_gap=2, min_hits=2)
    assert tracker.update([{'x': 10, 'y': 20}]) == []
    track = tracker.update([{'x': 12, 'y': 20}])[0]
    assert track.status == 'observed'
    track = tracker.update([])[0]
    assert track.status == 'predicted' and track.misses == 1
    assert tracker.update([])[0].misses == 2
    assert tracker.update([]) == []


def test_reacquisition_is_observed_not_ground_truth():
    tracker = MarkerTracker(gate_px=20, max_gap=2, min_hits=2)
    tracker.update([{'x': 10, 'y': 20}])
    original = tracker.update([{'x': 12, 'y': 20}])[0].id
    tracker.update([])
    actual = tracker.update([{'x': 15, 'y': 20}])[0]
    assert actual.id == original and actual.status == 'observed' and actual.misses == 0


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        MarkerTracker(max_gap=-1)
