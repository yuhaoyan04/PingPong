"""Unvalidated calibration experiment on geometry-filtered preview images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def run(items, train, test):
    obj = np.zeros((12, 3), np.float32)
    obj[:, :2] = np.asarray([(x * .05, y * .05) for y in range(4) for x in range(3)], np.float32)
    objs = [obj.copy() for _ in train]
    pts = lambda ix, role: [np.asarray(items[i]['views'][role]['points_px'], np.float32).reshape(-1, 1, 2) for i in ix]
    lm, rs = pts(train, 'master'), pts(train, 'slave')
    size = (1280, 720)
    e1, k1, d1, _, _ = cv2.calibrateCamera(objs, lm, size, None, None)
    e2, k2, d2, _, _ = cv2.calibrateCamera(objs, rs, size, None, None)
    rms, k1, d1, k2, d2, rotation, translation, _, _ = cv2.stereoCalibrate(
        objs, lm, rs, k1, d1, k2, d2, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 80, 1e-8))
    errors = []
    for i in test:
        left, right = pts([i], 'master')[0], pts([i], 'slave')[0]
        ok, rv, tv = cv2.solvePnP(obj, left, k1, d1)
        if not ok:
            raise RuntimeError('solvePnP failed')
        board_r, _ = cv2.Rodrigues(rv)
        other_r, _ = cv2.Rodrigues(rotation @ board_r)
        predicted, _ = cv2.projectPoints(obj, other_r, rotation @ tv + translation, k2, d2)
        delta = np.linalg.norm(predicted.reshape(-1, 2) - right.reshape(-1, 2), axis=1)
        errors.append({'pair': items[i]['pair'], 'rms_px': float(np.sqrt(np.mean(delta ** 2))),
                       'median_px': float(np.median(delta))})
    return {'train_ids': [items[i]['pair'] for i in train], 'test_ids': [items[i]['pair'] for i in test],
            'mono_rms_px': [float(e1), float(e2)], 'stereo_rms_px': float(rms),
            'baseline_assuming_50mm_m': float(np.linalg.norm(translation)),
            'focal_px': [float(k1[0, 0]), float(k1[1, 1]), float(k2[0, 0]), float(k2[1, 1])],
            'heldout_cross_projection': errors}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('session', type=Path)
    p.add_argument('geometry', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    items = json.loads((args.session / 'detection_audit.json').read_text(encoding='utf-8'))
    geometry = json.loads(args.geometry.read_text(encoding='utf-8'))
    selected = [i for i, g in enumerate(geometry) if g['pair'] != 39 and all(
        g['views'].get(role, {}).get('homography_rms_px', float('inf')) < 1.5 for role in ('master', 'slave'))]
    if len(selected) < 10:
        raise RuntimeError(f'Only {len(selected)} candidate pairs')
    experiments = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        order = rng.permutation(selected).tolist()
        experiments.append(run(items, order[3:], order[:3]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'warning': 'Diagnostic only; unknown scale, unvalidated correspondences, sampling and sync. Not for measurement.',
              'selection_homography_rms_lt_px': 1.5, 'candidate_ids': [items[i]['pair'] for i in selected],
              'experiments': experiments}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
