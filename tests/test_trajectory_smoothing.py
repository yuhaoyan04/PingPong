import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pingpong_event_pipeline.trajectory_smoothing import local_cubic, smooth_run
from pingpong_event_pipeline.marker_spin import fit_pairs, make_pairs


@pytest.mark.parametrize('rps',[0,5,20,40,80])
def test_curves_are_preserved_and_shape_jitter_is_reduced(rps):
    times=np.arange(31)*500
    phase=times*1e-6*rps*2*np.pi
    truth=np.column_stack([128+45*np.cos(phase),128+30*np.sin(phase)])
    clean,_,_=local_cubic(times,truth)
    assert np.max(np.linalg.norm(clean-truth,axis=1))<.1
    noise=truth+np.random.default_rng(31).normal(0,.6,truth.shape)
    noise[15]+=[-5,4]
    smooth,weights,status=local_cubic(times,noise)
    assert status['status']=='smoothed'
    assert np.mean((smooth-truth)**2)<.3*np.mean((noise-truth)**2)
    assert weights[15]<.3


def test_actual_curve_turn_is_not_forced_into_monotonic_motion():
    t=np.arange(15)*500+2**53+1
    x=np.arange(15)-7
    p=np.column_stack([128+2*x,128+x*x*.3])
    smooth,_,_=local_cubic(t,p)
    np.testing.assert_allclose(smooth,p,atol=1e-10)
    assert np.diff(smooth[:,1])[0]<0 and np.diff(smooth[:,1])[-1]>0


def test_short_segments_and_excessive_corrections_remain_original():
    p=np.array([[100,100],[110,105],[105,115]],float)
    q,_,status=local_cubic([0,500,1000],p)
    np.testing.assert_array_equal(p,q)
    assert status['status']=='short_segment_unchanged'
    p=np.column_stack([np.arange(15),np.zeros(15)]).astype(float);p[7,1]=50
    q,_,status=local_cubic(np.arange(15)*500,p)
    np.testing.assert_array_equal(p,q)
    assert status['status']=='large_correction_segment_unchanged'


@pytest.mark.parametrize('rps',[0,20,40,80])
def test_smoothing_preserves_known_sphere_rotation_and_signed_axis(rps):
    axis=np.array([.3,-.2,.9]);axis/=np.linalg.norm(axis)
    omega=axis*rps*2*np.pi
    all_runs=[]
    for k,xy in enumerate([[-.2,-.2],[.3,-.15],[-.2,.35]]):
        start=np.r_[xy,-np.sqrt(1-np.dot(xy,xy))]
        run=[]
        for i in range(15):
            sphere=Rotation.from_rotvec(omega*i*.0005).apply(start)
            # High-rate test uses shorter visible runs where needed.
            if sphere[2]>=-.15:break
            p=128+sphere[:2]*106.6666667
            run.append(dict(timestamp=10000+i*500,frame=i,point=p,sphere=sphere,radius=106.6666667,
                            source='detection_high',last_detection_confidence=.95,
                            x1=p[0]-5,y1=p[1]-5,x2=p[0]+5,y2=p[1]+5,
                            position_estimate_method='test',track_support='ring_supported'))
        if len(run)>=7:
            smoothed,_=smooth_run(run)
            assert [r['timestamp'] for r in run]==[r['timestamp'] for r in smoothed]
            all_runs.append((str(k),smoothed))
    pairs=make_pairs(all_runs,10000,20000)
    fit=fit_pairs(pairs,{'hypotheses':100})
    assert fit['numerically_usable']
    assert fit['candidate_rps']==pytest.approx(rps,abs=.15,rel=.025)
    if rps:assert np.dot(np.array(fit['omega_camera_rad_s'])/np.linalg.norm(fit['omega_camera_rad_s']),axis)>.995


def test_non_monotonic_times_are_rejected_before_filtering():
    with pytest.raises(ValueError,match='increasing'):
        local_cubic([0,1000,500],np.zeros((3,2)))


def test_unsupported_geometry_breaks_smoothing_context_and_no_observations_are_added():
    run=[]
    for i in range(17):
        p=np.array([110+i*.5,100.+np.sin(i)])
        run.append(dict(timestamp=i*500,point=p,radius=106.6667,source='detection_high',
                        last_detection_confidence=.95,x1=p[0]-4,y1=p[1]-4,x2=p[0]+4,y2=p[1]+4,
                        position_estimate_method='test',track_support='fallback_unverified' if i==8 else 'ring_supported'))
    whole,_=smooth_run(run);left,_=smooth_run(run[:8]);right,_=smooth_run(run[9:])
    assert len(whole)==len(run)
    np.testing.assert_array_equal(whole[8]['point'],run[8]['point'])
    assert not whole[8]['trajectory_smoothing_applied']
    np.testing.assert_allclose([r['point'] for r in whole[:8]],[r['point'] for r in left])
    np.testing.assert_allclose([r['point'] for r in whole[9:]],[r['point'] for r in right])
    with pytest.raises(ValueError,match='already smoothed'):
        smooth_run(left)


def test_radius_sensitivity_cases_use_identical_smoothed_pixel_coordinates():
    run=[]
    for i in range(12):
        p=np.array([100.+i,120.+np.sin(i*.3)])
        run.append(dict(timestamp=i*500,point=p,radius=106.6667,nominal_radius=106.6667,
                        source='detection_high',last_detection_confidence=.95,
                        x1=p[0]-4,y1=p[1]-4,x2=p[0]+4,y2=p[1]+4,
                        position_estimate_method='test',track_support='ring_supported'))
    base,_=smooth_run(run)
    variant,_=smooth_run([{**r,'radius':160.} for r in run])
    np.testing.assert_array_equal([r['point'] for r in base],[r['point'] for r in variant])
    assert not np.allclose([r['sphere'] for r in base],[r['sphere'] for r in variant])
