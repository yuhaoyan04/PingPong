import numpy as np
import pytest

from pingpong_event_pipeline.measurement_gate import image_motion_summary


def test_known_image_velocity_stays_unaccepted():
    result = image_motion_summary([0, 1000, 2000], [[20, 30], [21, 30], [22, 30]],
                                  [5, 5, 5], [1, 1, 1])
    assert result['median_image_speed_px_per_s'] == pytest.approx(1000)
    assert result['metric_speed_m_per_s'] is None and result['spin_rpm'] is None
    assert result['accepted'] is False


def test_reject_nonmonotonic_and_report_extrapolation():
    with pytest.raises(ValueError):
        image_motion_summary([0, 0, 2000], np.ones((3, 2)), [1, 1, 1], [1, 1, 1])
    result = image_motion_summary([0, 1000, 2000], [[20, 30], [21, 30], [22, 30]],
                                  [5, 5, 5], [1, 0, 1])
    assert result['measured_detection_fraction'] == pytest.approx(2/3)
