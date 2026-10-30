"""Sanity-check edge-normal sphere/translation geometry on known motion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_weighted_triplets import evaluate
from probe_stereo_normal_spin import fit_and_score


def test_known_rotation_beats_drift_with_perfect_normal_flow(tmp_path):
    rng = np.random.default_rng(12)
    n = 400
    uv = rng.uniform(-.5, .5, size=(n, 2))
    normal = rng.normal(size=(n, 2))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    omega = np.array([30., -20., 15.])
    drift = np.array([3., -2.])
    z = np.sqrt(1 - np.sum(uv*uv, axis=1))
    surface_flow = np.column_stack((z*omega[1]-uv[:,1]*omega[2],
                                    -z*omega[0]+uv[:,0]*omega[2]))
    raw_speed_signed = 50 * np.sum(normal*(surface_flow+drift),axis=1)
    raw_flow = normal * raw_speed_signed[:,None]
    timestamps = np.linspace(1000, 100000, n).astype(np.int64)
    path = tmp_path / 'path.npz'
    np.savez(path, timestamps_us=np.array([0, 101000]),
             centers_px=np.zeros((2,2)), radii_px=np.array([50.,50.]))
    result = evaluate(uv[:,0]*50, uv[:,1]*50, timestamps,
                      raw_flow[:,0], raw_flow[:,1],
                      np.full(n,3), np.zeros(n), path)
    assert result['status'] == 'diagnostic_only'
    assert result['early_to_late_improvement_over_drift'] > .999
    assert abs(result['early_rpm_candidate_unvalidated']
               -np.linalg.norm(omega)*60/(2*np.pi)) < 1e-6


def test_stereo_shared_spin_with_independent_drifts():
    rng = np.random.default_rng(21)
    angle = np.deg2rad(18)
    R = np.array([[np.cos(angle),-np.sin(angle),0],
                  [np.sin(angle),np.cos(angle),0], [0,0,1.]])
    world_omega = np.array([30.,-20.,15.])

    def camera_data(omega, nuisance):
        uv = rng.uniform(-.6,.6,size=(600,2))
        normal = rng.normal(size=(600,2))
        normal /= np.linalg.norm(normal,axis=1)[:,None]
        z = np.sqrt(1-np.sum(uv*uv,axis=1))
        nx,ny = normal.T
        x,y = uv.T
        design = np.column_stack((-ny*z,nx*z,-nx*y+ny*x,nx,ny))
        target = design @ np.r_[omega,nuisance]
        return design,design[:,3:],target,np.arange(len(target)),{'accepted_flows':len(target)}

    master = camera_data(world_omega,np.array([2.,-3.]))
    slave = camera_data(R@world_omega,np.array([-1.,2.]))
    result = fit_and_score(master,slave,R)
    assert result['status']=='diagnostic_only'
    assert min(result['heldout_improvement_over_independent_drift'].values())>.999
    assert result['early_late_axis_angle_deg']<1e-5
    assert abs(result['early_rpm_candidate_unvalidated']
               -np.linalg.norm(world_omega)*60/(2*np.pi))<1e-5
