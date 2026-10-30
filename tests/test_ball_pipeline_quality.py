import numpy as np
import pytest

from pingpong_event_pipeline.ball_pipeline_quality import assess_candidate


def test_support_gaps_are_not_counted_as_continuity():
    rows = np.zeros((22, 8))
    rows[:, 0] = np.arange(22)*1000
    rows[:, 1] = np.arange(22)*2
    rows[:, 5] = 1
    rows[10, 5] = 0
    report = assess_candidate(rows, coarse_count=10)
    assert report['status'] == 'unverified_candidate'
    assert report['longest_supported_run_ms'] == 11
    assert 'short_contiguous_support' in report['diagnostic_flags']


def test_rejects_unordered_or_missing_track():
    with pytest.raises(ValueError):
        assess_candidate(np.zeros((2, 8)), 2)
