"""Offline, CPU-only approximation of weighted event triplet matching.

Unlike the earlier nearest-triplet probe, this uses original integer pixels,
keeps several prior events per polarity/pixel, and averages every admissible
triplet with a temporal-consistency weight. It is still not the Metavision
implementation and must pass negative controls before any spin interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask


@njit(cache=True)
def weighted_triplets(x,y,t,p,width,height,radius,dt_min,dt_max,stride):
    depth = 4
    slots = width*height*2
    history = np.full((slots,depth),-10**15,np.int64)
    cursor = np.zeros(slots,np.int8)
    nout = (len(t)+stride-1)//stride
    vx = np.full(nout,np.nan)
    vy = np.full(nout,np.nan)
    count = np.zeros(nout,np.int32)
    spread = np.full(nout,np.nan)
    for k in range(len(t)):
        xx=int(x[k]); yy=int(y[k]); pp=int(p[k]); tk=int(t[k])
        if k%stride==0:
            sw=0.;sx=0.;sy=0.;s2=0.;found=0
            for oy in range(-radius,radius+1):
                yi=yy+oy
                if yi<0 or yi>=height:continue
                for ox in range(-radius,radius+1):
                    if ox==0 and oy==0:continue
                    if ox*ox+oy*oy>radius*radius:continue
                    xi=xx+ox
                    if xi<0 or xi>=width:continue
                    xj=xx+2*ox;yj=yy+2*oy
                    if xj<0 or xj>=width or yj<0 or yj>=height:continue
                    ik=(yi*width+xi)*2+pp
                    jk=(yj*width+xj)*2+pp
                    for a in range(depth):
                        ti=history[ik,a]
                        delta=tk-ti
                        if delta<dt_min or delta>dt_max:continue
                        expected=ti-delta
                        for b in range(depth):
                            tj=history[jk,b]
                            delta2=ti-tj
                            if delta2<dt_min or delta2>dt_max:continue
                            timing_error=abs(tj-expected)
                            if timing_error>0.3*delta:continue
                            sigma=max(50.,0.2*delta)
                            w=np.exp(-0.5*(timing_error/sigma)**2)
                            sec=(tk-tj)*1e-6
                            ux=(-2.*ox)/sec
                            uy=(-2.*oy)/sec
                            sw+=w;sx+=w*ux;sy+=w*uy
                            s2+=w*(ux*ux+uy*uy)
                            found+=1
            slot=k//stride
            if sw>0:
                vx[slot]=sx/sw;vy[slot]=sy/sw
                count[slot]=found
                spread[slot]=np.sqrt(max(0.,s2/sw-vx[slot]**2-vy[slot]**2))
        key=(yy*width+xx)*2+pp
        c=cursor[key]
        history[key,c]=tk
        cursor[key]=(c+1)%depth
    return vx,vy,count,spread


def build_normal_constraints(x,y,t,raw_vx,raw_vy,count,spread,path_file):
    path=np.load(path_file)
    pt=path['timestamps_us'].astype(float)
    centers=path['centers_px'].astype(float)
    radii=path['radii_px'].astype(float)
    cv=np.gradient(centers,pt*1e-6,axis=0)
    rv=np.gradient(radii,pt*1e-6)
    cx=np.interp(t,pt,centers[:,0]);cy=np.interp(t,pt,centers[:,1])
    r=np.interp(t,pt,radii)
    ux=(x-cx)/r;uy=(y-cy)/r
    raw_speed=np.hypot(raw_vx,raw_vy)
    good=(count>=3)&np.isfinite(raw_speed)&(raw_speed>1)&(raw_speed<50000)&(spread<5000)
    ux,uy,tt=ux[good],uy[good],t[good]
    nx,ny=raw_vx[good]/raw_speed[good],raw_vy[good]/raw_speed[good]
    centre_x=np.interp(tt,pt,cv[:,0]);centre_y=np.interp(tt,pt,cv[:,1])
    radius_dot=np.interp(tt,pt,rv)
    # Triplet flow is an edge-normal measurement. Subtract only the
    # projection of translation/expansion onto that same normal.
    target=(raw_speed[good]-nx*centre_x-ny*centre_y
            -radius_dot*(nx*ux+ny*uy))/r[good]
    z=np.sqrt(np.maximum(1e-8,1-ux*ux-uy*uy))
    full=np.column_stack((-ny*z,nx*z,-nx*uy+ny*ux,nx,ny))
    drift=full[:,3:]
    return full,drift,target,tt,good,raw_speed


def evaluate(x,y,t,raw_vx,raw_vy,count,spread,path_file):
    full,drift,target,tt,good,raw_speed=build_normal_constraints(
        x,y,t,raw_vx,raw_vy,count,spread,path_file)
    n=len(tt)
    if n<50:return {'accepted_flows':n,'status':'too_few_flows'}
    order=np.argsort(tt);train=order[:n//2];test=order[n//2:]
    q=np.linalg.lstsq(full[train],target[train],rcond=None)[0]
    q_late=np.linalg.lstsq(full[test],target[test],rcond=None)[0]
    d=np.linalg.lstsq(drift[train],target[train],rcond=None)[0]
    err=np.abs(full[test]@q-target[test])
    baseline=np.abs(drift[test]@d-target[test])
    qnorm=np.linalg.norm(q[:3]);late_norm=np.linalg.norm(q_late[:3])
    angle=np.degrees(np.arccos(np.clip(np.dot(q[:3],q_late[:3])/
        max(qnorm*late_norm,1e-12),-1,1)))
    return {'accepted_flows':n,'status':'diagnostic_only',
            'early_to_late_improvement_over_drift':float(1-np.median(err)/max(np.median(baseline),1e-9)),
            'full_model_condition':float(np.linalg.cond(full[train])),
            'early_rotation_candidate_rad_s_unvalidated':q[:3].tolist(),
            'early_rpm_candidate_unvalidated':float(qnorm*60/(2*np.pi)),
            'late_rpm_candidate_unvalidated':float(late_norm*60/(2*np.pi)),
            'early_late_axis_angle_deg':float(angle),
            'median_matched_triplets':float(np.median(count[good])),
            'median_raw_flow_px_s':float(np.median(np.hypot(raw_vx[good],raw_vy[good])))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True);p.add_argument('--path',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--inner-radius',type=float,default=.65)
    p.add_argument('--filter-threshold-us',type=int,default=5000)
    p.add_argument('--search-radius-px',type=int,default=5)
    p.add_argument('--dt-min-us',type=int,default=120)
    p.add_argument('--dt-max-us',type=int,default=3000)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--shuffle-event-positions',action='store_true')
    args=p.parse_args()
    rx,ry,t,pol,u,v,inside=load_ball_events(args.raw,args.path,args.inner_radius)
    filt=stc_cut_trail_mask(rx,ry,t,pol,int(rx.max())+1,int(ry.max())+1,args.filter_threshold_us)
    keep=inside&filt
    rx,ry,t,pol=rx[keep],ry[keep],t[keep],pol[keep]
    if args.shuffle_event_positions:
        rng=np.random.default_rng(100)
        perm=rng.permutation(len(t));rx,ry=rx[perm],ry[perm]
    vx,vy,count,spread=weighted_triplets(rx,ry,t,pol,1280,720,
            args.search_radius_px,args.dt_min_us,args.dt_max_us,args.stride)
    take=np.arange(0,len(t),args.stride)
    result={'parameters':vars(args),'filtered_ball_events':len(t),
            'anchors':len(take),'anchors_with_triplets':int(np.count_nonzero(count)),
            **evaluate(rx[take],ry[take],t[take],vx,vy,count,spread,args.path)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
