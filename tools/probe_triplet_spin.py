"""Small offline triplet-flow feasibility probe for DVSLume ball events.

This is deliberately a diagnostic, not a physical RPM estimator. It asks
whether short, constant-velocity event triplets span enough independent
directions to constrain one rigid sphere rotation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from estimate_event_plane_spin import normalized_events
from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask

def probe(x,y,t,p,seed=42,max_events=30000,max_anchors=2500,inner_radius=.65):
    rng=np.random.default_rng(seed)
    # Work in ball-radius-normalized pixels; 50 pixels represent one radius.
    if max_events > 0 and len(t)>max_events:
        ids=np.sort(rng.choice(len(t),max_events,replace=False))
        x,y,t,p=x[ids],y[ids],t[ids],p[ids]
    order=np.argsort(t,kind='stable');x,y,t,p=x[order],y[order],t[order],p[order]
    xy=np.column_stack((x,y))*50.0
    time=t.astype(float)
    anchors=np.linspace(0,len(t)-1,min(max_anchors,len(t)),dtype=int)
    # Time is scaled solely for neighbour search, then exact timestamp checks
    # impose the motion model.
    cloud=np.column_stack((xy,time*.006))
    trees=[cKDTree(cloud[p==pol]) for pol in (0,1)]
    subsets=[np.flatnonzero(p==pol) for pol in (0,1)]
    records=[];candidate_pairs=0
    for i in anchors:
        pol=int(p[i]);tree=trees[pol];sub=subsets[pol]
        neighbours=tree.query_ball_point(cloud[i],r=24.)
        first=[]
        for local in neighbours:
            j=int(sub[local]);dt=time[i]-time[j]
            if not 120<=dt<=2300:continue
            step=xy[i]-xy[j];distance=np.linalg.norm(step)
            if not 1.2<=distance<=9.:continue
            speed=distance/(dt*1e-6)
            if not 700<=speed<=50000:continue
            first.append((j,dt,step))
        candidate_pairs+=len(first)
        if len(first)>16:first=first[:16]
        best=None
        for j,dt,step in first:
            target=np.r_[xy[j]-step,(time[j]-dt)*.006]
            distance,local=tree.query(target,k=1)
            if not np.isfinite(distance):continue
            k=int(sub[local]);dt2=time[j]-time[k]
            if not .7*dt<=dt2<=1.3*dt:continue
            residual=np.linalg.norm(xy[k]-(xy[j]-step))
            if residual>1.3:continue
            score=residual+abs(dt2-dt)/dt
            if best is None or score<best[0]:best=(score,j,k,dt,step)
        if best is not None:
            _,j,k,dt,step=best
            vx,vy=step/50/(dt*1e-6)
            xx,yy=x[i],y[i];z=np.sqrt(max(1e-6,1-xx*xx-yy*yy))
            records.append((xx,yy,z,vx,vy,dt,time[i]))
    return np.asarray(records,dtype=float).reshape(-1,7),candidate_pairs,len(anchors)

def diagnose(records):
    if len(records)<10:return {'status':'insufficient_triplets'}
    x,y,z,vx,vy,dt,event_time=records.T
    A=np.zeros((2*len(records),3));A[0::2,1]=z;A[0::2,2]=-y
    A[1::2,0]=-z;A[1::2,2]=x
    target=np.column_stack((vx,vy)).ravel()
    rng=np.random.default_rng(37);perm=rng.permutation(len(records))
    train=perm[:len(perm)//2];held=perm[len(perm)//2:]
    rows=np.r_[2*train,2*train+1]
    q=np.linalg.lstsq(A[rows],target[rows],rcond=None)[0]
    held_rows=np.r_[2*held,2*held+1]
    residual=np.linalg.norm((A[held_rows]@q-target[held_rows]).reshape(-1,2),axis=1)
    zero=np.linalg.norm(target[held_rows].reshape(-1,2),axis=1)
    # A remaining error in the estimated ball centre appears as 2-D drift.
    # The sphere model must beat this simpler explanation on unseen time data.
    drift=np.median(np.column_stack((vx[train],vy[train])),axis=0)
    drift_residual=np.linalg.norm(np.column_stack((vx[held],vy[held]))-drift,axis=1)
    chronological=np.argsort(event_time)
    half=len(records)//2
    early=chronological[:half];late=chronological[half:]
    early_rows=np.r_[2*early,2*early+1]
    late_rows=np.r_[2*late,2*late+1]
    q_early=np.linalg.lstsq(A[early_rows],target[early_rows],rcond=None)[0]
    q_late=np.linalg.lstsq(A[late_rows],target[late_rows],rcond=None)[0]
    time_residual=np.linalg.norm((A[late_rows]@q_early-target[late_rows]).reshape(-1,2),axis=1)
    time_drift=np.median(np.column_stack((vx[early],vy[early])),axis=0)
    time_drift_residual=np.linalg.norm(np.column_stack((vx[late],vy[late]))-time_drift,axis=1)
    early_rpm=np.linalg.norm(q_early)*60/(2*np.pi)
    late_rpm=np.linalg.norm(q_late)*60/(2*np.pi)
    axis_cos=float(np.dot(q_early,q_late)/max(np.linalg.norm(q_early)*np.linalg.norm(q_late),1e-12))
    singular=np.linalg.svd(A[rows],compute_uv=False)
    bins=np.histogram2d(x,y,bins=4,range=[[-.65,.65],[-.65,.65]])[0]
    return {'status':'diagnostic_only','rpm_candidate_unvalidated':float(np.linalg.norm(q)*60/(2*np.pi)),
            'axis_camera':(q/np.linalg.norm(q)).tolist() if np.linalg.norm(q)>0 else None,
            'condition_number':float(singular[0]/max(singular[-1],1e-12)),
            'occupied_spatial_cells_4x4':int(np.count_nonzero(bins)),
            'heldout_median_residual_normalized_s':float(np.median(residual)),
            'heldout_median_zero_model_normalized_s':float(np.median(zero)),
            'heldout_improvement':float(1-np.median(residual)/max(np.median(zero),1e-9)),
            'heldout_improvement_over_constant_drift':float(1-np.median(residual)/max(np.median(drift_residual),1e-9)),
            'early_rpm_candidate':float(early_rpm),
            'late_rpm_candidate':float(late_rpm),
            'early_late_axis_angle_deg':float(np.degrees(np.arccos(np.clip(axis_cos,-1,1)))),
            'early_to_late_improvement_over_constant_drift':float(1-np.median(time_residual)/max(np.median(time_drift_residual),1e-9)),
            'median_triplet_delta_us':float(np.median(dt))}

def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--path',required=True);a.add_argument('--output',required=True)
    a.add_argument('--inner-radius',type=float,default=.65);a.add_argument('--seed',type=int,default=42)
    a.add_argument('--filter-threshold-us',type=int,default=0)
    a.add_argument('--max-events',type=int,default=30000)
    a.add_argument('--max-anchors',type=int,default=2500)
    a.add_argument('--shuffle-event-positions',action='store_true');args=a.parse_args()
    if args.filter_threshold_us:
        raw_x,raw_y,t,p,x,y,inside=load_ball_events(args.raw,args.path,args.inner_radius)
        mask=stc_cut_trail_mask(raw_x,raw_y,t,p,int(raw_x.max())+1,int(raw_y.max())+1,args.filter_threshold_us)
        selected=inside & mask
        x,y,t,p=x[selected],y[selected],t[selected],p[selected]
    else:
        x,y,t,p=normalized_events(args.raw,args.path,args.inner_radius)
    if args.shuffle_event_positions:
        perm=np.random.default_rng(args.seed+100).permutation(len(t))
        x,y=x[perm],y[perm]
    rec,pairs,anchors=probe(x,y,t,p,args.seed,max_events=args.max_events,
                            max_anchors=args.max_anchors,inner_radius=args.inner_radius)
    result={'events_inside_ball':len(t),'sampled_anchors':anchors,'candidate_pairs':pairs,
            'valid_triplets':len(rec),'triplet_fraction':len(rec)/max(anchors,1),**diagnose(rec),'parameters':vars(args)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
