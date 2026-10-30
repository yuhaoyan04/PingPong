"""Quantify sphere-rotation versus residual 2-D drift identifiability.

This optimistic calculation assumes perfect two-component flow everywhere.
Real event data provide noisier edge-normal flow, so a poor condition here is
decisive evidence of difficulty, while a good condition is not proof of success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask


def centered_rotation_design(u, v):
    z = np.sqrt(np.maximum(0.0, 1 - u*u - v*v))
    n = len(u)
    rotation = np.zeros((2*n, 3), dtype=np.float64)
    rotation[0::2, 1] = z
    rotation[0::2, 2] = -v
    rotation[1::2, 0] = -z
    rotation[1::2, 2] = u
    centered = rotation.copy()
    centered[0::2] -= rotation[0::2].mean(axis=0)
    centered[1::2] -= rotation[1::2].mean(axis=0)
    return rotation, centered, z


def geometry(u, v, radius_px, duration_s=0.002):
    rotation, centered, z = centered_rotation_design(u, v)
    n = len(u)
    drift = np.zeros((2*n, 2), dtype=np.float64)
    drift[0::2, 0] = 1
    drift[1::2, 1] = 1
    full = np.column_stack((rotation, drift))
    # Remove the best constant x/y flow, representing ball-centre error.
    singular = np.linalg.svd(full.T @ full, compute_uv=False)
    intrinsic = np.linalg.svd(centered.T @ centered, compute_uv=False)
    spin_10rps = 20*np.pi
    residual_px = [float(np.sqrt(np.mean((centered[:, j]*spin_10rps)**2))
                   * radius_px * duration_s) for j in range(3)]
    return {'events': n,
            'full_flow_rotation_plus_drift_condition': float(np.sqrt(singular[0]/singular[-1])),
            'rotation_after_drift_eigenvalue_ratio': float(intrinsic[0]/intrinsic[-1]),
            '10rps_residual_rms_px_per_component_over_2ms': residual_px,
            'median_surface_depth_z': float(np.median(z))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--filter-threshold-us', type=int, default=5000)
    p.add_argument('--slave-raw')
    p.add_argument('--slave-path')
    p.add_argument('--calibration', default='calibration/stereo_calibration.npz')
    args = p.parse_args()
    path = np.load(args.path)
    median_radius = float(np.median(path['radii_px']))
    x, y, t, pol, u, v, _ = load_ball_events(args.raw, args.path, 1.0)
    filtered = stc_cut_trail_mask(x, y, t, pol, int(x.max())+1,
                                  int(y.max())+1, args.filter_threshold_us)
    results = {'median_ball_radius_px': median_radius, 'idealized_full_flow': {}}
    slave = None
    if args.slave_raw and args.slave_path:
        sx, sy, st, sp, su, sv, _ = load_ball_events(args.slave_raw, args.slave_path, 1.0)
        sf = stc_cut_trail_mask(sx, sy, st, sp, int(sx.max())+1,
                                int(sy.max())+1, args.filter_threshold_us)
        slave = su, sv, sf
        rotation_master_to_slave = np.load(args.calibration)['R']
        results['stereo_idealized_full_flow'] = {}
    for limit in (0.4, 0.65, 0.85):
        selected = filtered & (u*u+v*v <= limit*limit)
        results['idealized_full_flow'][str(limit)] = geometry(u[selected], v[selected], median_radius)
        if slave is not None:
            su, sv, sf = slave
            ss = sf & (su*su+sv*sv <= limit*limit)
            _, mc, _ = centered_rotation_design(u[selected], v[selected])
            _, sc, _ = centered_rotation_design(su[ss], sv[ss])
            gram = mc.T @ mc + rotation_master_to_slave.T @ (sc.T @ sc) @ rotation_master_to_slave
            eigen = np.linalg.eigvalsh(gram)
            results['stereo_idealized_full_flow'][str(limit)] = {
                'master_events': int(selected.sum()), 'slave_events': int(ss.sum()),
                'rotation_after_independent_camera_drifts_eigenvalue_ratio': float(eigen[-1]/eigen[0]),
                'weakest_master_axis': np.linalg.eigh(gram)[1][:, 0].tolist()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
