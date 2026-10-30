import numpy as np
import pytest

from pingpong_event_pipeline.spin_direction import describe_omega, summarize_directions, classify_in_reference


def window(i, axis=(1, 0, 0), rps=40):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    return {"numerically_usable": True, "candidate_axis_camera": a.tolist(),
            "candidate_spin_rpm": rps * 60, "window_id": str(i)}


def test_signed_axis_and_front_surface_direction():
    assert describe_omega([1, 0, 0])["front_motion_label"] == "down"
    assert describe_omega([-1, 0, 0])["front_motion_label"] == "up"
    assert describe_omega([0, 1, 0])["front_motion_label"] == "left"
    assert describe_omega([0, -1, 0])["front_motion_label"] == "right"
    assert describe_omega([0, 0, 1])["image_z_spin_sense"] == "clockwise"
    assert describe_omega([0, 0, -1])["image_z_spin_sense"] == "counterclockwise"
    assert describe_omega([0, 0, 1])["front_motion_label"] == "unresolved"


def test_opposite_spin_is_not_mistaken_for_same_axis_line():
    result = summarize_directions([window(0), window(1, (-1, 0, 0)), window(2)])
    assert not result["engineering_stable"]
    assert result["all_windows_max_axis_deviation_deg"] == pytest.approx(180)


def test_temporal_outlier_excluded_without_clipping_measurements():
    rows = [window(i) for i in range(5)] + [window(5, (0, 0, 1), 150)]
    result = summarize_directions(rows)
    assert result["engineering_stable"] and result["stable_windows"] == 5
    assert result["engineering_candidate_rps"] == 40
    assert len(result["excluded_windows"]) == 1
    assert rows[-1]["candidate_spin_rpm"] == 9000  # raw fit retained


def test_plausibility_reference_does_not_force_estimates_to_normal_range():
    result = summarize_directions([window(i, rps=150) for i in range(3)])
    assert result["engineering_candidate_rps"] == 150
    assert describe_omega([1, 2, 3])["world_spin_type"] is None


def test_one_or_two_windows_do_not_establish_direction_stability():
    for n in [1, 2]:
        result = summarize_directions([window(i) for i in range(n)])
        assert result["engineering_direction_status"] == "insufficient_windows"
        assert result["engineering_candidate_rps"] is None


def test_sport_labels_require_explicit_reference_frame_and_correct_cross_product():
    forward, up = [0, 1, 0], [0, 0, 1]
    assert classify_in_reference([-1, 0, 0], forward, up)["world_spin_type"] == "topspin"
    assert classify_in_reference([1, 0, 0], forward, up)["world_spin_type"] == "backspin"
    assert classify_in_reference([0, 0, 1], forward, up)["world_spin_type"] == "side_ccw_from_above"
    assert classify_in_reference([0, 1, 0], forward, up)["world_spin_type"] == "predominantly_corkscrew"
    with pytest.raises(ValueError, match="parallel"):
        classify_in_reference([1, 0, 0], forward, forward)
