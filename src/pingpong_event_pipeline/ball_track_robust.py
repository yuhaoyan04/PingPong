"""Select a moving ball hypothesis from video circle candidates.

This is an offline hypothesis generator, not a measurement or a confidence guarantee.
"""
from __future__ import annotations

import numpy as np


def select_track(frames, min_speed=8.0, max_speed=90.0, gate=18.0,
                 max_gap=2, max_span=40, min_hits=5, radius_ratio=1.35):
    """Find a coherent approximately linear moving segment, allowing short gaps.

    Candidate tuples are x, y, radius, ring score. Units are pixels per 5 ms
    sample. Returned points are only detections; gaps are not fabricated.
    """
    if not (0 < min_speed < max_speed and gate > 0 and max_gap >= 0
            and max_span >= 2 and min_hits >= 3 and radius_ratio > 1):
        raise ValueError('invalid tracking parameters')
    arrays = [np.asarray(frame, dtype=float).reshape(-1, 4) for frame in frames]
    best = None
    for start, starts in enumerate(arrays):
        if not len(starts):
            continue
        for delta in range(1, min(5, len(arrays)-start)):
            ends = arrays[start+delta]
            if not len(ends):
                continue
            for a in starts[:5]:
                for b in ends[:5]:
                    velocity = (b[:2]-a[:2])/delta
                    speed = np.linalg.norm(velocity)
                    if not min_speed <= speed <= max_speed:
                        continue
                    if max(a[2], b[2]) / min(a[2], b[2]) > radius_ratio:
                        continue
                    members, distances, last = [], [], None
                    tracked_velocity = velocity.copy()
                    last_position = a[:2].copy()
                    for i in range(start, min(len(arrays), start+max_span)):
                        points = arrays[i]
                        if len(points):
                            pred = a[:2] if last is None else last_position + tracked_velocity*(i-last)
                            d = np.linalg.norm(points[:, :2] - pred, axis=1)
                            ratio = np.maximum(points[:, 2], a[2]) / np.minimum(points[:, 2], a[2])
                            d[ratio > radius_ratio] = np.inf
                            k = int(np.argmin(d))
                            if d[k] <= gate:
                                if last is not None and i-last > max_gap+1:
                                    break
                                if last is not None:
                                    observed_velocity = (points[k, :2] - last_position) / (i-last)
                                    tracked_velocity = .7*tracked_velocity + .3*observed_velocity
                                last_position = points[k, :2].copy()
                                members.append((i, points[k].copy()))
                                distances.append(float(d[k])); last = i
                        if last is not None and i-last > max_gap:
                            break
                    if len(members) < min_hits:
                        continue
                    idx = np.asarray([v[0] for v in members], float)
                    xy = np.asarray([v[1][:2] for v in members])
                    if np.linalg.norm(xy[-1] - xy[0]) / max(idx[-1] - idx[0], 1) < min_speed * .65:
                        continue
                    fit = np.polyfit(idx-idx[0], xy, 1)
                    residual = np.linalg.norm(xy - np.polyval(fit[:, 0], idx-idx[0])[:, None] * np.array([1., 0.])
                                              - np.polyval(fit[:, 1], idx-idx[0])[:, None] * np.array([0., 1.]), axis=1)
                    median_error = float(np.median(residual))
                    ring = float(np.mean([v[1][3] for v in members]))
                    # Keep a visible cost for interpolated gaps and incoherent fits.
                    gaps = int(idx[-1]-idx[0]+1-len(idx))
                    score = len(members) * (0.5 + ring) - .22*gaps - .15*median_error
                    if best is None or score > best['score']:
                        best = {'members': members, 'score': float(score), 'median_fit_error_px': median_error,
                                'mean_ring_score': ring, 'gap_count': gaps, 'speed_px_per_step': float(speed)}
    return best
