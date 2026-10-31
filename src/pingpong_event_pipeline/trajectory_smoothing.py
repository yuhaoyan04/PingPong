"""Robust local cubic smoothing of evidence-backed marker-centre segments.

Uses actual event timestamps and two-sided offline context. No spin model,
monotonic-coordinate constraint, gap filling or cross-segment information.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class SmoothingConfig:
    enabled: bool = True
    half_window_us: int = 1500
    minimum_samples: int = 7
    degree: int = 3
    iterations: int = 8
    noise_floor_px: float = .5
    huber_k: float = 1.5
    shape_log_sigma: float = .4
    maximum_correction_px: float = 8.

    def __post_init__(self):
        if self.half_window_us<=0 or self.minimum_samples<7 or self.degree!=3 or self.iterations<1:
            raise ValueError('require positive window, >=7 observations, cubic degree and positive iterations')
        if not all(np.isfinite(v) and v>0 for v in [self.noise_floor_px,self.huber_k,self.shape_log_sigma,self.maximum_correction_px]):
            raise ValueError('smoothing scales must be finite and positive')


def local_cubic(times_us, points, quality=None, config=None):
    cfg=config if isinstance(config,SmoothingConfig) else SmoothingConfig(**(config or {}))
    times=np.asarray(times_us,np.int64);points=np.asarray(points,float)
    if points.shape!=(len(times),2) or not np.isfinite(points).all() or np.any(np.diff(times)<=0):
        raise ValueError('need finite Nx2 points and strictly increasing integer timestamps')
    quality=np.ones(len(times)) if quality is None else np.asarray(quality,float)
    if quality.shape!=(len(times),) or not np.isfinite(quality).all() or np.any(quality<=0):
        raise ValueError('quality weights must be finite and positive')
    if not cfg.enabled or len(times)<cfg.minimum_samples:
        return points.copy(),np.ones(len(times)),{'status':'disabled' if not cfg.enabled else 'short_segment_unchanged'}
    # Subtract integers before conversion, preserving absolute timestamp precision.
    t=(times-times[0]).astype(float);out=np.empty_like(points);own_weights=np.ones(len(times))
    for i,now in enumerate(t):
        indices=np.flatnonzero(np.abs(t-now)<=cfg.half_window_us)
        if len(indices)<cfg.minimum_samples:
            indices=np.sort(np.argsort(np.abs(t-now),kind='stable')[:cfg.minimum_samples])
        scale=max(cfg.half_window_us,float(np.max(np.abs(t[indices]-now))))
        x=(t[indices]-now)/scale
        design=np.column_stack([x**k for k in range(cfg.degree+1)])
        # Weak distance weighting; never zero out one-sided endpoint support.
        base=quality[indices]*np.exp(-.5*x*x)
        weights=base.copy();target=points[indices];robust=np.ones(len(indices))
        for _ in range(cfg.iterations):
            root=np.sqrt(weights);a=design*root[:,None]
            coefficients=np.linalg.lstsq(a,target*root[:,None],rcond=None)[0]
            residual=target-design@coefficients
            # Correct residuals for leverage: an endpoint must not validate itself.
            hat=np.sum(a*(a@np.linalg.pinv(a.T@a)),axis=1)
            radius=np.linalg.norm(residual,axis=1)/np.sqrt(np.maximum(1-hat,.05))
            med=np.median(residual,axis=0)
            sigma=max(cfg.noise_floor_px,float(1.4826*np.median(np.abs(residual-med))))
            robust=np.minimum(1,cfg.huber_k*sigma/np.maximum(radius,1e-12))
            next_weights=base*robust
            if np.max(np.abs(next_weights-weights))<1e-5:
                weights=next_weights;break
            weights=next_weights
        root=np.sqrt(weights)
        coefficients=np.linalg.lstsq(design*root[:,None],target*root[:,None],rcond=None)[0]
        out[i]=coefficients[0]
        own_weights[i]=float(robust[np.flatnonzero(indices==i)[0]])
    correction=np.linalg.norm(out-points,axis=1)
    if not np.isfinite(out).all() or correction.max()>cfg.maximum_correction_px:
        return points.copy(),own_weights,{'status':'large_correction_segment_unchanged','proposed_maximum_correction_px':float(correction.max())}
    return out,own_weights,{'status':'smoothed','maximum_correction_px':float(correction.max())}


def smooth_run(run, config=None):
    """Return same observations/IDs/times, preserving original bounding boxes."""
    cfg=config if isinstance(config,SmoothingConfig) else SmoothingConfig(**(config or {}))
    if any(str(r.get('trajectory_smoothing_applied','')).lower()=='true' for r in run):
        raise ValueError('trajectory already smoothed; disable additional smoothing for this export')
    if any(r.get('track_support')!='ring_supported' for r in run):
        # A guessed ball circle cannot carry smoothing information into good
        # geometry on the other side. Keep its observations but break context.
        result=[];subreports=[]
        for r in run:
            result.append({**r,'raw_point':r['point'].copy(),
                           'raw_center_x_px':float(r['point'][0]),'raw_center_y_px':float(r['point'][1]),
                           'trajectory_smoothing_applied':False,'trajectory_smoothing_status':'unsupported_ball_geometry_unchanged',
                           'smoothing_residual_weight':1.,'smoothing_shape_weight':1.,'smoothing_displacement_px':0.})
        indices=np.array([i for i,r in enumerate(run) if r.get('track_support')=='ring_supported'],int)
        for part in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1):
            if not len(part):continue
            smoothed,report=smooth_run([run[i] for i in part],cfg)
            subreports.append(report)
            for i,r in zip(part,smoothed):result[int(i)]=r
        return result,{'status':'partially_smoothed' if any(r['trajectory_smoothing_applied'] for r in result) else 'no_supported_long_subsegment',
                       'config':asdict(cfg),'observations':len(run),'supported_subsegments':subreports}
    times=np.array([r['timestamp'] for r in run],np.int64)
    raw=np.array([r['point'] for r in run])
    sizes=np.array([[float(r['x2'])-float(r['x1']),float(r['y2'])-float(r['y1'])] for r in run])
    if not np.isfinite(sizes).all() or np.any(sizes<=0):raise ValueError('invalid tracker bounding box size')
    shape=np.ones(len(run));source_quality=np.ones(len(run))
    for i,r in enumerate(run):
        neighbors=np.abs(times-times[i])<=cfg.half_window_us
        median_size=np.median(sizes[neighbors],axis=0)
        shape[i]=max(.15,float(np.exp(-.5*np.mean((np.log(sizes[i]/median_size)/cfg.shape_log_sigma)**2))))
        # Confidence from a previous detector hit is stale for optical-flow frames.
        source_quality[i]=.5 if r['source']=='optical_flow' else .6 if r['source']=='detection_low' else max(.5,min(1,float(r['last_detection_confidence'])))
    smoothed,robust,summary=local_cubic(times,raw,shape*source_quality,cfg)
    # Keep the geometry guard fixed across radius sensitivity experiments.
    nominal=float(run[0].get('nominal_radius',run[0]['radius']))
    if np.any(np.linalg.norm(smoothed-128,axis=1)>=nominal):
        smoothed=raw.copy();summary={'status':'sphere_boundary_segment_unchanged'}
    applied=summary['status']=='smoothed'
    result=[]
    for i,r in enumerate(run):
        p=smoothed[i];xy=(p-128)/r['radius']
        if float(xy@xy)>=1:raise ValueError('smoothed point outside requested sphere')
        result.append({**r,'raw_point':raw[i].copy(),'point':p.copy(),
                       'sphere':np.r_[xy,-np.sqrt(1-xy@xy)],
                       'raw_center_x_px':float(raw[i,0]),'raw_center_y_px':float(raw[i,1]),
                       'center_x_px':float(p[0]),'center_y_px':float(p[1]),
                       'trajectory_smoothing_applied':applied,'trajectory_smoothing_status':summary['status'],
                       'smoothing_residual_weight':float(robust[i]),'smoothing_shape_weight':float(shape[i]),
                       'smoothing_displacement_px':float(np.linalg.norm(p-raw[i])),
                       'position_estimate_method':r['position_estimate_method']+(' + offline robust local cubic' if applied else '')})
    return result,{**summary,'config':asdict(cfg),'observations':len(run),
                   'downweighted_observations':int(np.count_nonzero(robust<.5)),
                   'median_correction_px':float(np.median(np.linalg.norm(smoothed-raw,axis=1)))}


def csv_rows(segments):
    return [{k:v for k,v in row.items() if k not in {'frame','timestamp','sphere','point','raw_point','radius','nominal_radius'}}
            for run in segments.values() for row in run]
