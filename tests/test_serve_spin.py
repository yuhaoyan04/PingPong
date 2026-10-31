import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pingpong_event_pipeline.serve_spin import (align_axes, cross_validate_orientation,
                                              fit_two_view, common_support, proper_rotation)
from test_yolo_marker_spin import synthetic_pairs, CFG


def test_joint_rotation_uses_known_camera_orientation_without_marker_id_matching():
    r = Rotation.from_euler("y", 35, degrees=True).as_matrix()
    omega = np.array([12., -8., 16.])
    pairs = {"F3E": synthetic_pairs(omega), "F59": synthetic_pairs(r.T @ omega)}
    fit = fit_two_view(pairs, r, CFG)
    assert fit["numerically_usable"]
    np.testing.assert_allclose(fit["omega_camera_rad_s"], omega, atol=1e-5)
    assert fit["candidate_rps"] == pytest.approx(np.linalg.norm(omega) / (2*np.pi), abs=1e-6)
    assert all(v["inlier_fraction"] == 1 for v in fit["per_view"].values())
    assert not fit["accepted"] and fit["spin_rpm"] is None


def test_joint_zero_and_missing_or_disjoint_observations_are_distinct():
    p = {c: synthetic_pairs((0., 0., 0.)) for c in ["F3E", "F59"]}
    fit = fit_two_view(p, np.eye(3), CFG)
    assert fit["numerically_usable"] and fit["candidate_rps"] == 0
    assert fit["candidate_axis_camera"] is None
    assert fit["rotation_signal"] == "near_zero_unresolved_direction"
    empty = fit_two_view({**p, "F59": []}, np.eye(3), CFG)
    assert empty["candidate_rps"] is None
    later = [(s, {**a, "timestamp": a["timestamp"] + 30000},
              {**b, "timestamp": b["timestamp"] + 30000}, dt) for s, a, b, dt in p["F59"]]
    disjoint = fit_two_view({**p, "F59": later}, np.eye(3), CFG)
    assert not disjoint["numerically_usable"]
    assert disjoint["rejection_reasons"] == ["insufficient_simultaneous_observations"]


def test_opposite_angular_velocities_cannot_be_fused_by_flipping_sign():
    p = {"F3E": synthetic_pairs((100., 0., 0.)), "F59": synthetic_pairs((-100., 0., 0.))}
    fit = fit_two_view(p, np.eye(3), CFG)
    assert not fit["numerically_usable"]
    assert "inconsistent_two_view_surface_motion" in fit["rejection_reasons"]


def test_one_corrupted_marker_in_one_view_is_rejected_without_losing_good_view():
    p = {"F3E": synthetic_pairs(), "F59": synthetic_pairs(corrupt=True)}
    fit = fit_two_view(p, np.eye(3), CFG)
    assert fit["numerically_usable"]
    np.testing.assert_allclose(fit["omega_camera_rad_s"], [40, -35, 100], atol=1.)
    assert fit["per_view"]["F59"]["inlier_fraction"] < 1


def registration_samples():
    r = Rotation.from_euler("xyz", [10, 165, 4], degrees=True).as_matrix()
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -.4, .3]], float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return r, [{"serve": f"serve{i}", "axis_F3E": a.tolist(), "axis_F59": (r.T @ a).tolist()} for i, a in enumerate(axes)]


def test_relative_orientation_is_cross_validated_by_whole_serve_without_leakage():
    expected, samples = registration_samples()
    result = cross_validate_orientation(samples)
    assert result["usable"] and result["leave_one_serve_out_passed"]
    np.testing.assert_allclose(result["R_F59_to_F3E"], expected, atol=1e-6)
    for held in result["leave_one_serve_out"]:
        assert held["held_out_serve"] not in held["registration"]["training_serves"]
        assert held["angle_error_deg"] < 1e-4
    assert not result["measured_extrinsics"]


def test_parallel_axes_or_too_few_serves_do_not_establish_orientation():
    _, samples = registration_samples()
    assert not align_axes(samples[:2])["usable"]
    parallel = [{"serve": str(i), "axis_F3E": [1, 0, 0], "axis_F59": [-1, 0, 0]} for i in range(5)]
    assert not cross_validate_orientation(parallel)["usable"]
    with pytest.raises(ValueError, match="proper rotation"):
        proper_rotation(np.diag([-1., 1., 1.]))


def test_single_view_fallback_keeps_alternatives_and_rejects_conflicting_rates():
    from pingpong_event_pipeline.serve_review import conclusion
    from pingpong_event_pipeline.spin_direction import summarize_directions
    from test_spin_direction import window
    views = {c: {"aligned_summary": summarize_directions([window(i,rps=rate) for i in range(n)]),
                 "supported_observed_frames": 50} for c, rate, n in [("F3E",37,3),("F59",42,5)]}
    serve = {"joint_summary": summarize_directions([]), "views": views, "R_F59_to_F3E": np.eye(3).tolist()}
    result = conclusion(serve)
    assert result["mode"] == "single_view_fallback" and result["camera"] == "F59"
    assert result["rps"] == 42 and len(result["single_view_alternatives"]) == 2
    views['F59']['aligned_summary']['engineering_candidate_rps'] = 90
    assert conclusion(serve)['rps'] is None
