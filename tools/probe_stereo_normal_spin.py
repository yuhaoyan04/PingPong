"""Test one shared spin vector against independent drift in both cameras.

Uses the experimental weighted-triplet normal-flow probe. All fitted angular
velocities are diagnostic only until time, stereo, parameter, and null-model
gates pass; this script intentionally does not export a measured RPM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask
from probe_weighted_triplets import build_normal_constraints, weighted_triplets


def camera(raw,path,inner_radius,threshold_us,search_radius,stride):
    x,y,t,p,_,_,inside=load_ball_events(raw,path,inner_radius)
    filtered=stc_cut_trail_mask(x,y,t,p,int(x.max())+1,int(y.max())+1,threshold_us)
    keep=inside&filtered
    x,y,t,p=x[keep],y[keep],t[keep],p[keep]
    vx,vy,count,spread=weighted_triplets(x,y,t,p,1280,720,search_radius,120,3000,stride)
    take=np.arange(0,len(t),stride)
    full,drift,target,time,good,speed=build_normal_constraints(
        x[take],y[take],t[take],vx,vy,count,spread,path)
    return full,drift,target,time,{'ball_events':len(t),'accepted_flows':len(time)}


def fit_and_score(master,slave,R):
    mf,md,my,mt,ms=master
    sf,sd,sy,st,ss=slave
    if min(len(my),len(sy))<100:
        return {'status':'too_few_flows','master':ms,'slave':ss}
    # R transforms an angular velocity in master camera coordinates into
    # slave camera coordinates; each camera gets its own 2-D drift nuisance.
    m=np.column_stack((mf[:,:3],md,np.zeros_like(md)))
    s=np.column_stack((sf[:,:3]@R,np.zeros_like(sd),sd))
    baseline_m=m[:,3:]
    baseline_s=s[:,3:]
    mi=np.argsort(mt);si=np.argsort(st)
    me,ml=mi[:len(mi)//2],mi[len(mi)//2:]
    se,sl=si[:len(si)//2],si[len(si)//2:]
    # Equal camera weight, despite different numbers of accepted events.
    mw=1/np.sqrt(len(me));sw=1/np.sqrt(len(se))
    train=np.vstack((m[me]*mw,s[se]*sw))
    truth=np.r_[my[me]*mw,sy[se]*sw]
    q=np.linalg.lstsq(train,truth,rcond=None)[0]
    q_late=np.linalg.lstsq(
        np.vstack((m[ml]/np.sqrt(len(ml)),s[sl]/np.sqrt(len(sl)))),
        np.r_[my[ml]/np.sqrt(len(ml)),sy[sl]/np.sqrt(len(sl))],rcond=None)[0]
    d=np.linalg.lstsq(train[:,3:],truth,rcond=None)[0]
    measurements=[]
    for design,output,late,name in ((m,my,ml,'master'),(s,sy,sl,'slave')):
        model=np.median(np.abs(design[late]@q-output[late]))
        null=np.median(np.abs(design[late,3:]@d-output[late]))
        measurements.append((name,float(1-model/max(null,1e-12)),float(model),float(null)))
    magnitude=np.linalg.norm(q[:3]);late_magnitude=np.linalg.norm(q_late[:3])
    angle=float(np.degrees(np.arccos(np.clip(
        np.dot(q[:3],q_late[:3])/max(magnitude*late_magnitude,1e-12),-1,1))))
    return {'status':'diagnostic_only','master':ms,'slave':ss,
            'heldout_improvement_over_independent_drift':{name:value for name,value,_,_ in measurements},
            'pooled_camera_median_improvement':float(np.mean([row[1] for row in measurements])),
            'early_late_axis_angle_deg':angle,
            'early_rpm_candidate_unvalidated':float(magnitude*60/(2*np.pi)),
            'late_rpm_candidate_unvalidated':float(late_magnitude*60/(2*np.pi)),
            'joint_condition':float(np.linalg.cond(train))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master-raw',required=True);p.add_argument('--master-path',required=True)
    p.add_argument('--slave-raw',required=True);p.add_argument('--slave-path',required=True)
    p.add_argument('--calibration',default='calibration/stereo_calibration.npz')
    p.add_argument('--output',required=True)
    p.add_argument('--inner-radius',type=float,default=.65)
    p.add_argument('--filter-threshold-us',type=int,default=5000)
    p.add_argument('--search-radius-px',type=int,default=5)
    p.add_argument('--stride',type=int,default=5)
    args=p.parse_args()
    m=camera(args.master_raw,args.master_path,args.inner_radius,args.filter_threshold_us,
             args.search_radius_px,args.stride)
    s=camera(args.slave_raw,args.slave_path,args.inner_radius,args.filter_threshold_us,
             args.search_radius_px,args.stride)
    R=np.load(args.calibration)['R']
    result={'parameters':vars(args),**fit_and_score(m,s,R)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
