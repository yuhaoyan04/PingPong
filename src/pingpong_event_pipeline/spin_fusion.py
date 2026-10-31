"""Time-local spin selection and camera-correct surface motion for review.

Keep independent camera estimates distinct from a shared SO(3) estimate. A
single-camera fallback is explicitly labeled; conflicting fallbacks stay unknown.
"""
from __future__ import annotations
import numpy as np
from .spin_direction import describe_omega
from .spin import _rotate_vectors

CAMERAS=('F3E','F59')


def time_window_id(serve,timestamp_us):
    candidates=[w for w in serve['windows'] if w['start_us']<=timestamp_us<w['end_us']]
    chosen=min(candidates,key=lambda w:(abs((w['start_us']+w['end_us'])/2-timestamp_us),w['start_us']),default=None)
    return chosen['window_id'] if chosen else None


def fit_at(fit,timestamp_us):
    return bool(fit.get('numerically_usable') and timestamp_us in fit.get('observation_timestamps_us',()))


def camera_omega(omega,reference,camera,rotation):
    w=np.asarray(omega,float)
    if camera==reference:return w.copy()
    if rotation is None:return None
    r=np.asarray(rotation,float)
    if r.shape!=(3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1):
        raise ValueError('camera mapping must be a proper rotation')
    return r.T@w if reference=='F3E' else r@w


def public_fit(fit,reference):
    near_zero=fit.get('rotation_signal')=='near_zero_unresolved_direction'
    w=np.array(fit['omega_camera_rad_s'],float);magnitude=np.linalg.norm(w)
    return {'available':True,'rps':float(fit['candidate_rps']),'rpm':float(fit['candidate_spin_rpm']),
            'omega_camera_rad_s':w.tolist(),'reference_camera':reference,
            'axis_camera':(w/magnitude).tolist() if magnitude>1e-8 and not near_zero else None,
            'direction_resolved':bool(not near_zero and magnitude>1e-8),
            'direction':describe_omega(np.zeros(3) if near_zero else w),
            'inlier_fraction':fit.get('inlier_fraction',fit.get('balanced_inlier_fraction'))}


def absent(reason='no_current_observations'):
    return {'available':False,'rps':None,'rpm':None,'omega_camera_rad_s':None,'reference_camera':None,
            'axis_camera':None,'direction_resolved':False,'direction':None,'reason':reason}


def frame_spin(serve,timestamp_us,window_id=None):
    """One time-selected window; current observations only, with no value hold."""
    window_id=time_window_id(serve,timestamp_us) if window_id is None else window_id
    window=next((w for w in serve['windows'] if w['window_id']==window_id),None)
    result={'timestamp_us':int(timestamp_us),'window_id':window_id,'independent':{},'system':absent(),
            'system_camera_directions':{},'warnings':[]}
    for c in CAMERAS:
        fit=window['mono'][c] if window else {}
        result['independent'][c]=public_fit(fit,c) if fit_at(fit,timestamp_us) else absent()
    if window is None:return result
    available=[c for c in CAMERAS if result['independent'][c]['available']]
    if len(available)==2:
        x,y=[result['independent'][c] for c in CAMERAS]
        difference=abs(x['rps']-y['rps'])
        if difference>max(1.,.20*(x['rps']+y['rps'])/2):result['warnings'].append('independent_rate_disagreement')
        if x['direction_resolved'] and y['direction_resolved'] and serve.get('R_F59_to_F3E') is not None:
            a=np.array(x['axis_camera']);b=np.asarray(serve['R_F59_to_F3E'])@np.array(y['axis_camera'])
            angle=float(np.degrees(np.arccos(np.clip(a@b,-1,1))))
            result['independent_axis_disagreement_degrees']=angle
            if angle>20:result['warnings'].append('independent_axis_disagreement')
    if fit_at(window,timestamp_us):
        # Joint RANSAC can resolve a bad independent consensus. Keep its own
        # estimate and expose any disagreement rather than averaging scalars.
        result['system']={**public_fit(window,'F3E'),'source':'JOINT','selection':'shared_SO3_fit'}
    elif result['warnings']:
        result['system']={**absent('conflicting_independent_estimates'),'source':None,'selection':'unknown'}
    elif available:
        c=max(available,key=lambda c:(window['mono'][c].get('inlier_fraction',0),
                                     window['mono'][c].get('unique_observations',0),
                                     window['mono'][c].get('duration_us',0)))
        result['system']={**result['independent'][c],'source':c,'selection':'explicit_single_camera_fallback'}
    else:result['system']={**absent(),'source':None,'selection':'unknown'}
    system=result['system']
    if system['available']:
        for c in CAMERAS:
            local=camera_omega(system['omega_camera_rad_s'],system['reference_camera'],c,serve.get('R_F59_to_F3E'))
            result['system_camera_directions'][c]={'omega_camera_rad_s':local.tolist(),
                'direction':describe_omega(local if system['direction_resolved'] else np.zeros(3))} if local is not None else None
    return result


def marker_motion(sphere,omega,radius_px,horizon_us=500):
    """Finite projected trajectory at this marker, including the view-axis term.

    This is modeled surface motion, not an added observed point. Return both a
    current tangent and a future endpoint so draw code can make the distinction.
    """
    s=np.asarray(sphere,float);w=np.asarray(omega,float)
    if s.shape!=(3,) or w.shape!=(3,) or not np.isfinite([*s,*w,radius_px,horizon_us]).all():raise ValueError('finite sphere/omega/radius/horizon required')
    if radius_px<=0 or horizon_us<=0 or abs(np.linalg.norm(s)-1)>1e-5 or s[2]>1e-8:raise ValueError('require visible unit sphere and positive scales')
    velocity=np.cross(w,s)[:2]*radius_px
    future=_rotate_vectors(s[None],w,np.array([horizon_us*1e-6]))[0]
    displacement=(future[:2]-s[:2])*radius_px
    speed=float(np.linalg.norm(velocity));resolved=bool(speed*horizon_us*1e-6>=.25 and future[2]<=0)
    return {'resolved':resolved,'tangent_direction':(velocity/speed).tolist() if resolved else None,
            'velocity_px_s':velocity.tolist(),'predicted_displacement_px':displacement.tolist(),
            'horizon_us':int(horizon_us),'future_visible':bool(future[2]<=0)}


def check_motion_pair(sphere0,sphere1,dt_s,omega,radius_px):
    """Suppress misleading local arrows; never flip omega to match a bad point."""
    a=np.asarray(sphere0,float);b=np.asarray(sphere1,float)
    if not np.isfinite(dt_s) or dt_s<=0:raise ValueError('positive actual pair duration required')
    predicted=_rotate_vectors(a[None],np.asarray(omega,float),np.array([dt_s]))[0]
    actual=(b[:2]-a[:2])*radius_px;model=(predicted[:2]-a[:2])*radius_px
    angular=float(np.arccos(np.clip(predicted@b,-1,1)))
    na,nb=np.linalg.norm(actual),np.linalg.norm(model)
    angle=float(np.degrees(np.arccos(np.clip(actual@model/(na*nb),-1,1)))) if min(na,nb)>1e-9 else None
    reason=None
    if angular>.035:reason='outside_rotation_consensus'
    elif na<.75 or nb<.25:reason='local_motion_too_small'
    elif angle is None or angle>45:reason='local_direction_disagreement'
    return {'accepted':reason is None,'reason':reason,'direction_error_degrees':angle,
            'angular_error_rad':angular,'observed_displacement_px':actual.tolist(),
            'predicted_displacement_px':model.tolist(),'actual_pair_dt_us':round(dt_s*1e6)}
