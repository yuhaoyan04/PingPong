"""Diagnostic local event-plane normal-flow versus residual drift.

This is not the vendor Triplet Matching implementation and emits no RPM.
It tests whether local spatiotemporal planes favour one sphere rotation over
the simpler explanation of 2-D ball-centre drift on time-held-out events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask


def collect(x, y, t, polarity, max_anchors, seed, max_plane_residual_us):
    rng = np.random.default_rng(seed)
    order = np.argsort(t, kind='stable')
    x, y, t, polarity = x[order], y[order], t[order], polarity[order]
    n = len(t)
    anchors = np.linspace(0, n-1, min(max_anchors, n)).astype(int)
    coords = np.column_stack((x*50, y*50, (t-t[0])*.006))
    groups = [np.flatnonzero(polarity == p) for p in (0, 1)]
    trees = [cKDTree(coords[g]) for g in groups]
    records = []
    plane_residuals = []
    reasons = {'few_neighbours': 0, 'poor_geometry': 0,
               'poor_plane': 0, 'speed_outside_range': 0}
    for i in anchors:
        g = groups[int(polarity[i])]
        neighbours = g[trees[int(polarity[i])].query_ball_point(coords[i], r=7.0)]
        du = (x[neighbours]-x[i])*50
        dv = (y[neighbours]-y[i])*50
        dt = t[neighbours]-t[i]
        keep = (np.abs(dt) <= 900) & (du*du+dv*dv <= 25)
        du, dv, dt = du[keep], dv[keep], dt[keep]
        if len(dt) < 12:
            reasons['few_neighbours'] += 1
            continue
        design = np.column_stack((du, dv, np.ones(len(dt))))
        if np.linalg.cond(design) > 25:
            reasons['poor_geometry'] += 1
            continue
        a, b, c = np.linalg.lstsq(design, dt, rcond=None)[0]
        residual = float(np.median(np.abs(design @ [a,b,c]-dt)))
        plane_residuals.append(residual)
        if residual > max_plane_residual_us:
            reasons['poor_plane'] += 1
            continue
        gradient = np.hypot(a,b)
        if not 50 <= gradient <= 2500:
            reasons['speed_outside_range'] += 1
            continue
        # Seconds per ball radius; plane equation a*vx+b*vy = 1.
        a *= 50e-6
        b *= 50e-6
        z = np.sqrt(max(1e-9, 1-x[i]*x[i]-y[i]*y[i]))
        records.append((x[i],y[i],z,a,b,t[i],residual))
    quantiles = np.percentile(plane_residuals, [1,5,10,25,50,75]).tolist() if plane_residuals else []
    return np.asarray(records, dtype=float).reshape(-1,7), reasons, len(anchors), quantiles


def evaluate(records):
    if len(records) < 50:
        return {'status':'too_few_normal_flows'}
    x,y,z,a,b,time,plane_residual = records.T
    rotation = np.column_stack((-b*z,a*z,-a*y+b*x))
    drift = np.column_stack((a,b))
    full = np.column_stack((rotation,drift))
    truth = np.ones(len(records))
    order = np.argsort(time)
    early = order[:len(order)//2]
    late = order[len(order)//2:]
    model = np.linalg.lstsq(full[early], truth[early], rcond=None)[0]
    drift_only = np.linalg.lstsq(drift[early], truth[early], rcond=None)[0]
    model_residual = np.abs(full[late] @ model - truth[late])
    drift_residual = np.abs(drift[late] @ drift_only - truth[late])
    singular = np.linalg.svd(full[early], compute_uv=False)
    return {'status':'diagnostic_only',
            'median_plane_residual_us':float(np.median(plane_residual)),
            'early_to_late_median_full_residual':float(np.median(model_residual)),
            'early_to_late_median_drift_residual':float(np.median(drift_residual)),
            'early_to_late_improvement_over_drift':float(1-np.median(model_residual)/max(np.median(drift_residual),1e-12)),
            'full_model_condition':float(singular[0]/max(singular[-1],1e-12))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inner-radius', type=float, default=.65)
    p.add_argument('--filter-threshold-us', type=int, default=5000)
    p.add_argument('--max-anchors', type=int, default=1500)
    p.add_argument('--max-plane-residual-us', type=float, default=140)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-event-positions', action='store_true')
    args = p.parse_args()
    rx,ry,t,pol,x,y,inside = load_ball_events(args.raw,args.path,args.inner_radius)
    filtered = stc_cut_trail_mask(rx,ry,t,pol,int(rx.max())+1,int(ry.max())+1,args.filter_threshold_us)
    selected = inside & filtered
    x,y,t,pol=x[selected],y[selected],t[selected],pol[selected]
    if args.shuffle_event_positions:
        perm = np.random.default_rng(args.seed+100).permutation(len(t))
        x,y=x[perm],y[perm]
    records,reasons,anchors,quantiles = collect(x,y,t,pol,args.max_anchors,args.seed,
                                                args.max_plane_residual_us)
    result={'parameters':vars(args),'ball_events':len(t),'sampled_anchors':anchors,
            'accepted_normal_flows':len(records),'rejected':reasons,
            'plane_residual_quantiles_1_5_10_25_50_75_us':quantiles,**evaluate(records)}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
