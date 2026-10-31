"""Evidence-gated ball circles and material-point localization, without spin priors.

Circle measurements use angularly balanced RAW edge samples. Marker centers are
transported from one anchor using local image registration, rather than following
the changing bounding-box midpoint. Rejected estimates retain the original.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


@dataclass
class CircleConfig:
    sectors: int=72
    minimum_sectors: int=36
    annulus_fraction: float=.28
    maximum_center_fraction: float=.18
    maximum_radius_fraction: float=.20
    inlier_px: float=1.5
    minimum_validation_improvement_px: float=.10
    maximum_condition: float=8.
    hypotheses: int=256


def circle_from_three(points):
    a=np.asarray(points,float)
    try:
        c=np.linalg.solve(2*(a[1:]-a[0]),np.sum(a[1:]**2,axis=1)-np.sum(a[0]**2))
    except np.linalg.LinAlgError:return None
    return np.r_[c,np.linalg.norm(a[0]-c)]


def edge_sectors(points,seed,weights=None,sectors=72,annulus_fraction=.28):
    """One radial density ridge per angular sector, not one vote per event."""
    xy=np.asarray(points,float).reshape(-1,2);seed=np.asarray(seed,float)
    w=np.ones(len(xy)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(xy),) or not np.isfinite(w).all() or np.any(w<0):raise ValueError('invalid event weights')
    if seed.shape!=(3,) or not np.isfinite(seed).all() or seed[2]<=0:raise ValueError('invalid circle seed')
    d=xy-seed[:2];r=np.linalg.norm(d,axis=1)
    keep=np.isfinite(xy).all(axis=1)&(abs(r-seed[2])<=seed[2]*annulus_fraction)&(w>0)
    xy=xy[keep];w=w[keep];d=d[keep];r=r[keep]
    if not len(xy):return np.empty((0,2)),np.empty(0,int)
    # Repeated firings at one pixel must not overwhelm independent boundary arcs.
    keys=np.rint(xy*2).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse);w=w/count[inverse]
    sector=np.floor((np.arctan2(d[:,1],d[:,0])+np.pi)*sectors/(2*np.pi)).astype(int)%sectors
    lower=seed[2]*(1-annulus_fraction);step=.5
    radial=np.floor((r-lower)/step).astype(int)
    bins=int(np.ceil(2*seed[2]*annulus_fraction/step))+1
    points_out=[];ids=[]
    for s in range(sectors):
        mask=sector==s
        if np.count_nonzero(mask)<4:continue
        profile=np.bincount(radial[mask],weights=w[mask],minlength=bins)
        smooth=gaussian_filter1d(profile,.9)
        peak=int(np.argmax(smooth));at=lower+(peak+.5)*step
        selected=mask&(abs(r-at)<=1.25)
        if np.count_nonzero(selected)<3 or smooth[peak]<.45:continue
        points_out.append(np.average(xy[selected],axis=0,weights=w[selected]));ids.append(s)
    return np.array(points_out,float).reshape(-1,2),np.array(ids,int)


def fit_boundary_circle(points,seed,weights=None,config=None):
    """Disjoint sectors for fit, acceptance, and untouched diagnostic residuals."""
    cfg=config or CircleConfig();seed=np.asarray(seed,float)
    xy,sectors=edge_sectors(points,seed,weights,cfg.sectors,cfg.annulus_fraction)
    result={'accepted':False,'circle':seed.tolist(),'proposed_circle':None,'sectors':len(xy),
            'status':'insufficient_boundary_sectors','config':asdict(cfg)}
    if len(xy)<cfg.minimum_sectors:return result
    train=sectors%3==0;valid=sectors%3==1;audit=sectors%3==2
    if min(train.sum(),valid.sum(),audit.sum())<10:return result
    if len(set((sectors*4//cfg.sectors).tolist()))<3:return result
    limit=seed[2]*cfg.maximum_center_fraction
    lo=np.r_[seed[:2]-limit,seed[2]*(1-cfg.maximum_radius_fraction)]
    hi=np.r_[seed[:2]+limit,seed[2]*(1+cfg.maximum_radius_fraction)]
    def residual(circle,pts):return np.linalg.norm(pts-circle[:2],axis=1)-circle[2]
    def solve(pts):
        rng=np.random.default_rng(181);candidates=[seed]
        for _ in range(cfg.hypotheses):
            c=circle_from_three(pts[rng.choice(len(pts),3,replace=False)])
            if c is not None and np.all(c>lo) and np.all(c<hi):candidates.append(c)
        def score(c):
            r=abs(residual(c,pts));return (int(np.sum(r<cfg.inlier_px)),-float(np.median(r)))
        c=max(candidates,key=score)
        for _ in range(2):
            mask=abs(residual(c,pts))<2.5*cfg.inlier_px
            if mask.sum()<8:break
            c=least_squares(lambda v:residual(v,pts[mask]),c,bounds=(lo,hi),loss='soft_l1',f_scale=.75,max_nfev=60).x
        return c
    proposed=solve(xy[train]);before=abs(residual(seed,xy[valid]));after=abs(residual(proposed,xy[valid]))
    unit=(xy[train]-proposed[:2])/np.maximum(np.linalg.norm(xy[train]-proposed[:2],axis=1,keepdims=True),1e-9)
    design=np.c_[unit,np.ones(train.sum())];condition=float(np.linalg.cond(design))
    reasons=[]
    if condition>cfg.maximum_condition:reasons.append('ill_conditioned_arcs')
    if np.median(before)-np.median(after)<cfg.minimum_validation_improvement_px:reasons.append('no_unseen_arc_improvement')
    if np.mean(after<=cfg.inlier_px)+.03<np.mean(before<=cfg.inlier_px):reasons.append('unseen_arc_support_decreased')
    if np.mean(after<=cfg.inlier_px)<.60:reasons.append('insufficient_unseen_arc_support')
    if np.any(proposed-lo<.05) or np.any(hi-proposed<.05):reasons.append('parameter_at_search_boundary')
    # The other half must independently predict a similar circle.
    other=solve(xy[valid]);disagreement=float(np.linalg.norm(proposed-other))
    if disagreement>1.5:reasons.append('independent_arc_fit_disagreement')
    result.update(status='accepted' if not reasons else ','.join(reasons),accepted=not reasons,
                  proposed_circle=proposed.tolist(),validation_before_median_px=float(np.median(before)),
                  validation_after_median_px=float(np.median(after)),validation_before_inlier_fraction=float(np.mean(before<=cfg.inlier_px)),
                  audit_before_median_px=float(np.median(abs(residual(seed,xy[audit])))),
                  audit_after_median_px=float(np.median(abs(residual(proposed,xy[audit])))),
                  validation_after_inlier_fraction=float(np.mean(after<=cfg.inlier_px)),condition=condition,
                  independent_arc_disagreement_px=disagreement,edge_points=xy.tolist(),edge_sector_ids=sectors.tolist())
    if not reasons:result['circle']=proposed.tolist()
    return result


@dataclass
class LocalizationConfig:
    minimum_features: int=12
    minimum_training_inliers: int=5
    maximum_forward_backward_px: float=.75
    maximum_validation_error_px: float=1.25
    minimum_validation_improvement_px: float=.05
    maximum_correction_px: float=8.
    minimum_scale: float=.75
    maximum_scale: float=1.30
    maximum_rotation_degrees: float=35.


def registration_image(polarity_rgb):
    image=np.asarray(polarity_rgb)
    if image.ndim!=3 or image.shape[2]!=3:raise ValueError('expected RGB polarity image')
    # Yellow=ON, blue=OFF; signed contrast retains the two lobes separately.
    value=128+(.5*image[:,:,0].astype(float)-.5*image[:,:,2].astype(float))
    return cv2.GaussianBlur(np.clip(value,0,255).astype(np.uint8),(0,0),.8)


def marker_features(image,box):
    h,w=image.shape;box=np.asarray(box,float)
    x1,y1=np.maximum(0,np.floor(box[:2]+1)).astype(int)
    x2,y2=np.minimum([w,h],np.ceil(box[2:]-1)).astype(int)
    mask=np.zeros(image.shape,np.uint8)
    if x2>x1 and y2>y1:mask[y1:y2,x1:x2]=255
    p=cv2.goodFeaturesToTrack(image,mask=mask,maxCorners=80,qualityLevel=.025,minDistance=2,blockSize=3)
    return np.empty((0,2),np.float32) if p is None else p.reshape(-1,2)


def register_marker(reference,current,reference_box,reference_center,current_center,config=None):
    """Transport an anchor center; spatially held-out corners gate affine motion."""
    cfg=config or LocalizationConfig();p0=marker_features(reference,reference_box)
    result={'accepted':False,'center':np.asarray(current_center,float).tolist(),'status':'insufficient_features','features':len(p0)}
    if len(p0)<cfg.minimum_features:return result
    delta=np.asarray(current_center)-np.asarray(reference_center)
    initial=(p0+delta).astype(np.float32).reshape(-1,1,2);p=p0.reshape(-1,1,2)
    options=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.005))
    forward,s1,_=cv2.calcOpticalFlowPyrLK(reference,current,p,initial,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
    if forward is None or not np.isfinite(forward).all():result['status']='forward_flow_failed';return result
    backward,s2,_=cv2.calcOpticalFlowPyrLK(current,reference,forward,p.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
    if backward is None:result['status']='backward_flow_failed';return result
    fb=np.linalg.norm(backward.reshape(-1,2)-p0,axis=1)
    good=(s1.ravel()>0)&(s2.ravel()>0)&np.isfinite(fb)&(fb<=cfg.maximum_forward_backward_px)
    a=p0[good];b=forward.reshape(-1,2)[good]
    result['valid_features']=len(a)
    if len(a)<cfg.minimum_features:result['status']='insufficient_consistent_features';return result
    # Separate fit, acceptance, and audit corners. Audit errors do not choose
    # the transform or the accept/reject decision.
    order=np.lexsort((a[:,1],a[:,0]));hold=np.zeros(len(a),bool);hold[order[::4]]=True
    audit=np.zeros(len(a),bool);audit[order[1::4]]=True;train=~(hold|audit)
    cv2.setRNGSeed(181)
    affine,inliers=cv2.estimateAffine2D(a[train],b[train],method=cv2.RANSAC,ransacReprojThreshold=1.0,maxIters=1000,confidence=.99,refineIters=10)
    if affine is None or not np.isfinite(affine).all():result['status']='affine_fit_failed';return result
    result['training_inliers']=int(inliers.sum())
    if inliers.sum()<cfg.minimum_training_inliers:result['status']='insufficient_affine_inliers';return result
    predicted=a[hold]@affine[:,:2].T+affine[:,2]
    before=np.linalg.norm(a[hold]+delta-b[hold],axis=1)
    after=np.linalg.norm(predicted-b[hold],axis=1)
    point=affine[:,:2]@np.asarray(reference_center)+affine[:,2]
    scale=np.linalg.svd(affine[:,:2],compute_uv=False)
    angle=abs(float(np.degrees(np.arctan2(affine[1,0]-affine[0,1],affine[0,0]+affine[1,1]))))
    covariance=np.cov(a[train][inliers.ravel()>0].T)
    reasons=[]
    if np.linalg.det(affine[:,:2])<=0 or scale.min()<cfg.minimum_scale or scale.max()>cfg.maximum_scale:reasons.append('excessive_deformation')
    if angle>cfg.maximum_rotation_degrees:reasons.append('excessive_local_rotation')
    if np.linalg.eigvalsh(covariance).min()<2.:reasons.append('insufficient_feature_spread')
    if np.median(after)>cfg.maximum_validation_error_px:reasons.append('heldout_feature_error')
    if np.median(before)-np.median(after)<cfg.minimum_validation_improvement_px:reasons.append('no_heldout_feature_improvement')
    if np.linalg.norm(point-current_center)>cfg.maximum_correction_px:reasons.append('large_center_correction')
    result.update(status='accepted' if not reasons else ','.join(reasons),accepted=not reasons,
                  proposed_center=point.tolist(),affine=affine.tolist(),validation_features=int(hold.sum()),
                  validation_before_median_px=float(np.median(before)),validation_after_median_px=float(np.median(after)),
                  audit_features=int(audit.sum()),
                  audit_before_median_px=float(np.median(np.linalg.norm(a[audit]+delta-b[audit],axis=1))),
                  audit_after_median_px=float(np.median(np.linalg.norm(a[audit]@affine[:,:2].T+affine[:,2]-b[audit],axis=1))),
                  median_forward_backward_px=float(np.median(fb[good])),center_correction_px=float(np.linalg.norm(point-current_center)),
                  scale_singular_values=scale.tolist(),local_rotation_degrees=angle)
    if not reasons:result['center']=point.tolist()
    return result
