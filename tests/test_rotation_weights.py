import numpy as np
import pytest
from pingpong_event_pipeline.rotation_weights import correspondence_weights,weighted_soft_l1
from pingpong_event_pipeline.marker_spin import fit_pairs
from pingpong_event_pipeline.serve_spin import fit_two_view
from test_yolo_marker_spin import synthetic_pairs,CFG


def replicated_corrupt_pairs():
    pairs=synthetic_pairs(corrupt=True)
    bad=[p for p in pairs if p[0]=='0'];good=[p for p in pairs if p[0]!='0']
    return good+bad*30


def test_equal_cluster_mass_is_invariant_to_repeated_correspondences():
    pairs=replicated_corrupt_pairs();w=correspondence_weights(pairs,'cluster_equal')
    masses=[sum(v for p,v in zip(pairs,w) if p[0]==key) for key in ['0','1','2','3']]
    np.testing.assert_allclose(masses,np.repeat(sum(w)/4,4),rtol=1e-12)
    fit=fit_pairs(pairs,{**CFG,'hypotheses':512,'pair_weighting':'cluster_equal'})
    assert fit['numerically_usable']
    np.testing.assert_allclose(fit['omega_camera_rad_s'],[40,-35,100],atol=1)
    # The corrupt track contains both valid and invalid lags. Its many repeated
    # invalid lags must lose influence without pretending the whole track is bad.
    assert fit['weighted_inlier_fraction']>fit['inlier_fraction']+.2
    assert fit['weighted_inlier_fraction']>=.7


def test_joint_equal_camera_and_cluster_weighting_resists_one_long_bad_track():
    fit=fit_two_view({'F3E':synthetic_pairs(),'F59':replicated_corrupt_pairs()},np.eye(3),
                     {**CFG,'hypotheses':512,'pair_weighting':'cluster_equal'})
    assert fit['numerically_usable']
    np.testing.assert_allclose(fit['omega_camera_rad_s'],[40,-35,100],atol=1)
    assert fit['per_view']['F59']['inlier_fraction']>fit['per_view']['F59']['raw_pair_inlier_fraction']+.2
    assert fit['per_view']['F59']['inlier_fraction']>=.7


def test_zero_rotation_and_no_observations_stay_distinct_with_balancing():
    config={**CFG,'pair_weighting':'cluster_equal'}
    zero=fit_pairs(synthetic_pairs((0,0,0)),config)
    assert zero['numerically_usable'] and zero['candidate_rps']==0
    assert zero['candidate_axis_camera'] is None
    assert fit_pairs([],config)['candidate_rps'] is None


def test_weighted_robust_loss_derivatives_match_finite_differences():
    f=weighted_soft_l1([.2,1,3]);z=np.array([.3,2,12.]);eps=1e-5
    value=f(z);up=f(z+eps);down=f(z-eps)
    np.testing.assert_allclose(value[1],(up[0]-down[0])/(2*eps),rtol=1e-6)
    np.testing.assert_allclose(value[2],(up[1]-down[1])/(2*eps),rtol=1e-6)
    with pytest.raises(ValueError):correspondence_weights([],mode='not-a-mode')
