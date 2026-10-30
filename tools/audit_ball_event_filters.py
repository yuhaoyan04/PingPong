"""Audit pixel-level STC-CUT-TRAIL on recorded ball events.

This is a diagnostic implementation of the documented burst rule, not a
vendor-verified bit-for-bit reproduction of a camera's hardware filter.
Filtering occurs on the original sensor pixels before ball coordinates are
normalised or any events are sampled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

from pingpong_event_pipeline.events import RawWindowReader


@njit(cache=True)
def stc_cut_trail_mask(x, y, t, polarity, width, height, threshold_us):
    last_t = np.full(width * height, -10**15, dtype=np.int64)
    last_p = np.full(width * height, -1, dtype=np.int8)
    stage = np.zeros(width * height, dtype=np.int8)
    keep = np.zeros(len(t), dtype=np.bool_)
    for i in range(len(t)):
        k = int(y[i]) * width + int(x[i])
        if polarity[i] != last_p[k] or t[i] - last_t[k] > threshold_us:
            stage[k] = 1  # first event: suppress
        elif stage[k] == 1:
            stage[k] = 2  # second event: retain
            keep[i] = True
        last_p[k] = polarity[i]
        last_t[k] = t[i]
    return keep


def load_ball_events(raw, path_file, inner_radius):
    path = np.load(path_file)
    path_t = path['timestamps_us'].astype(np.int64)
    centres = path['centers_px'].astype(float)
    radii = path['radii_px'].astype(float)
    events = RawWindowReader(raw).window(int(path_t[0]), int(path_t[-1]) + 1)
    t = events['timestamp'].astype(np.int64)
    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    polarity = events['polarity'].astype(np.int8)
    cx = np.interp(t, path_t, centres[:, 0])
    cy = np.interp(t, path_t, centres[:, 1])
    radius = np.interp(t, path_t, radii)
    u = (x - cx) / radius
    v = (y - cy) / radius
    inside = u*u + v*v <= inner_radius*inner_radius
    return x, y, t, polarity, u, v, inside


def profile(u, v, t, selected, inner_radius):
    count = int(np.count_nonzero(selected))
    if not count:
        return {'events': 0, 'occupied_cells_8x8': 0,
                'occupied_5ms_bins': 0, 'events_per_5ms_bin_p10': 0}
    h = np.histogram2d(u[selected], v[selected], bins=8,
                       range=[[-inner_radius, inner_radius]]*2)[0]
    bins = np.bincount(((t[selected] - t.min()) // 5000).astype(np.int64))
    return {'events': count,
            'occupied_cells_8x8': int(np.count_nonzero(h)),
            'occupied_5ms_bins': int(np.count_nonzero(bins)),
            'events_per_5ms_bin_p10': float(np.percentile(bins, 10)),
            'events_per_5ms_bin_median': float(np.median(bins))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inner-radius', type=float, default=0.65)
    p.add_argument('--thresholds-us', type=int, nargs='+', default=[500, 2000, 5000])
    args = p.parse_args()
    x, y, t, pol, u, v, inside = load_ball_events(args.raw, args.path, args.inner_radius)
    width = int(x.max()) + 1
    height = int(y.max()) + 1
    result = {'raw': args.raw, 'path': args.path,
              'inner_radius': args.inner_radius,
              'sensor_extent_seen': [width, height],
              'all_window_events': len(t),
              'unfiltered_ball': profile(u, v, t, inside, args.inner_radius),
              'filters': {}}
    for threshold in args.thresholds_us:
        filtered = stc_cut_trail_mask(x, y, t, pol, width, height, threshold)
        selected = inside & filtered
        stats = profile(u, v, t, selected, args.inner_radius)
        stats['retention_fraction'] = stats['events'] / max(result['unfiltered_ball']['events'], 1)
        result['filters'][str(threshold)] = stats
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
