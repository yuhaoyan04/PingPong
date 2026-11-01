import copy
import json
import numpy as np
import pytest
from pingpong_event_pipeline.spin_fusion import frame_spin,time_window_id,marker_motion,camera_omega,check_motion_pair
from pingpong_event_pipeline.spin import _rotate_vectors
from test_radius_review import fixture_serve


def test_independent_values_are_not_copies_of_system_rate_or_scalar_averages():
    s=fixture_serve(40);w=s['windows'][0]
    for camera,rate in [('F3E',39),('F59',41)]:
        w['mono'][camera].update(candidate_rps=rate,candidate_spin_rpm=rate*60)
    v=frame_spin(s,100)
    assert v['independent']['F3E']['rps']==39 and v['independent']['F59']['rps']==41
    assert v['system']['rps']==40 and v['system']['source']=='JOINT'
    assert json.loads(json.dumps(v,allow_nan=False))['system']['rps']==40


def test_conflicting_mono_fallbacks_remain_visible_but_system_is_unknown():
    s=fixture_serve(40);w=s['windows'][0];w['numerically_usable']=False
    w['mono']['F59'].update(candidate_rps=100,candidate_spin_rpm=6000)
    v=frame_spin(s,100)
    assert v['independent']['F3E']['available'] and v['independent']['F59']['available']
    assert not v['system']['available'] and v['system']['rps'] is None
    assert 'independent_rate_disagreement' in v['warnings']


def test_one_camera_fallback_is_explicit_and_never_held_into_a_gap():
    s=fixture_serve(4);w=s['windows'][0];w['numerically_usable']=False;w['mono']['F59']['numerically_usable']=False
    v=frame_spin(s,100)
    assert v['system']['source']=='F3E' and v['system']['rps']==4
    assert not frame_spin(s,300)['system']['available']
    assert not frame_spin(s,1100)['system']['available']


def test_supported_zero_is_distinct_from_unknown_and_has_no_arbitrary_arrow():
    v=frame_spin(fixture_serve(0),100)
    assert v['system']['available'] and v['system']['rps']==0
    assert not v['system']['direction_resolved']
    assert not marker_motion([0,0,-1],[0,0,0],100)['resolved']


def test_window_selection_depends_on_time_and_does_not_chase_valid_or_lower_rates():
    s=fixture_serve();other=copy.deepcopy(s['windows'][0]);other.update(window_id='serve/500',start_us=500,end_us=1500)
    s['windows'].append(other);s['windows'][0]['numerically_usable']=False
    assert time_window_id(s,500)=='serve/0'
    assert time_window_id(s,1000)=='serve/500'


def test_current_overlapping_valid_fit_is_not_lost_at_failed_window_boundary():
    s=fixture_serve(40);other=copy.deepcopy(s['windows'][0]);other.update(window_id='serve/500',start_us=500,end_us=1500)
    for f in [s['windows'][0],*s['windows'][0]['mono'].values()]:f['observation_timestamps_us']=[750]
    for f in [other,*other['mono'].values()]:f.update(numerically_usable=False,observation_timestamps_us=[750])
    s['windows'].append(other)
    v=frame_spin(s,750,'serve/500')
    assert v['system']['available'] and v['system']['rps']==40
    assert v['system']['window_id']==s['windows'][0]['window_id']
    assert v['system']['overlap_window_fallback']
    assert not frame_spin(s,800,'serve/500')['system']['available']


def test_view_axis_spin_has_local_circular_motion_even_when_front_proxy_is_zero():
    w=[0,0,100]
    right=marker_motion([.5,0,-np.sqrt(.75)],w,100)
    left=marker_motion([-.5,0,-np.sqrt(.75)],w,100)
    assert right['tangent_direction']==pytest.approx([0,1])
    assert left['tangent_direction']==pytest.approx([0,-1])
    assert not marker_motion([0,0,-1],w,100)['resolved']


def test_camera_transform_preserves_handedness_and_sign():
    r=np.diag([-1.,1.,-1.])
    w=camera_omega([100,0,0],'F3E','F59',r)
    assert marker_motion([0,0,-1],w,100)['tangent_direction']==pytest.approx([0,-1])
    assert camera_omega([1,2,3],'F3E','F59',None) is None
    with pytest.raises(ValueError):camera_omega([1,2,3],'F3E','F59',np.diag([-1,1,1]))


def test_wrong_signed_local_motion_is_rejected_without_flipping_the_estimate():
    a=np.array([.3,.2,-np.sqrt(.87)]);w=np.array([100.,20.,-30.]);dt=.0005
    b=_rotate_vectors(a[None],w,np.array([dt]))[0]
    assert check_motion_pair(a,b,dt,w,100)['accepted']
    rejected=check_motion_pair(a,b,dt,-w,100)
    assert not rejected['accepted']
    assert rejected['direction_error_degrees']>170


def test_unresolved_subpixel_motion_does_not_get_a_prominent_direction_arrow():
    a=np.array([0.,0.,-1.]);b=_rotate_vectors(a[None],[.1,0,0],np.array([.0005]))[0]
    result=check_motion_pair(a,b,.0005,[.1,0,0],100)
    assert not result['accepted'] and result['reason']=='local_motion_too_small'
