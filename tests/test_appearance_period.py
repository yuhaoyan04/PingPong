import numpy as np
import pytest
from pingpong_event_pipeline.appearance_period import recurrence, common_peaks


def test_recurring_texture_preserves_fundamental_and_multiple_turn_branches():
    phase=np.arange(160)*2*np.pi/40
    features=np.column_stack([np.cos(phase),np.sin(phase)])
    result=recurrence(features,np.arange(160)*500)
    peaks=result['peaks']
    assert {p['lag_frames'] for p in peaks}>={40,80,120}
    first=next(p for p in peaks if p['lag_frames']==40)
    assert first['delay_us']==20000 and first['equivalent_1_over_delay_rps']==50
    common=common_peaks(peaks,peaks)
    assert len(common)>=3 and not any(p['physical_spin_accepted'] for p in common)


def test_missing_or_uninformative_texture_is_not_zero_spin():
    result=recurrence(np.zeros((30,8)),np.arange(30)*500)
    assert result['peaks']==[] and common_peaks([],[])==[]
    with pytest.raises(ValueError,match='uniform'):
        recurrence(np.zeros((4,8)),[0,500,1500,2000])
