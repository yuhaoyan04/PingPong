import numpy as np

from pingpong_event_pipeline.ball_track_robust import select_track


def test_prefers_fast_coherent_ball_over_long_slow_background():
    frames = []
    for i in range(32):
        frame = [(100+i*1.0, 100, 30, .8)]
        if 5 <= i <= 17 and i != 10:
            frame.insert(0, (300+i*2, 20+i*20, 38, .8))
        frames.append(frame)
    result = select_track(frames)
    assert result is not None
    assert len(result['members']) >= 11
    assert result['members'][0][1][0] > 300
    assert 10 not in [i for i, _ in result['members']]
    assert result['gap_count'] == 1


def test_stationary_noise_and_too_few_observations_are_rejected():
    assert select_track([[(100+i, 100, 30, .9)] for i in range(20)]) is None
    assert select_track([[(100+i*20, 100, 30, .9)] for i in range(3)]) is None
