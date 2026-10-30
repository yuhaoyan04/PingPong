"""Experimental local RAW-event circle refinement around a coarse ball track."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt, savgol_filter

from .detector import HoughCircleBallDetector


def robust_seed(points):
    """Fit a smooth seed path and flag spatial outliers; times are microseconds."""
    points = np.asarray(points, dtype=float).reshape(-1, 4)
    if len(points) < 4 or np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError('four or more ordered coarse detections are required')
    t = (points[:, 0] - points[0, 0]) / 1000.0
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        curves = [UnivariateSpline(t[keep], points[keep, axis], k=2, s=16*keep.sum()) for axis in (1, 2)]
        estimated = np.column_stack([curve(t) for curve in curves])
        error = np.linalg.norm(points[:, 1:3] - estimated, axis=1)
        median = np.median(error[keep])
        mad = np.median(np.abs(error[keep]-median))
        good = error <= max(8., median + 3.5*1.4826*mad)
        if good.sum() < 4 or np.array_equal(good, keep):
            break
        keep = good
    curves = [UnivariateSpline(t[keep], points[keep, axis], k=2, s=16*keep.sum()) for axis in (1, 2)]
    radius = medfilt(points[keep, 3], kernel_size=3)
    radius[0], radius[-1] = points[keep, 3][0], points[keep, 3][-1]
    return t, keep, curves, radius


def warped_local_image(events, end_us, center, velocity_px_us, half_size):
    """Render one event window after translation to its end, without polarity merging in RAW."""
    size = 2*half_size + 1
    image = np.zeros((size, size), np.uint8)
    if not len(events):
        return image
    age = end_us - events['timestamp'].astype(np.int64)
    valid = (age >= 0) & (age <= 1000)
    age = age[valid]
    x = np.rint(events['x'][valid].astype(float) + velocity_px_us[0]*age - center[0] + half_size).astype(int)
    y = np.rint(events['y'][valid].astype(float) + velocity_px_us[1]*age - center[1] + half_size).astype(int)
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    values = np.clip(255*np.exp(-age[inside]/500.), 0, 255).astype(np.uint8)
    np.maximum.at(image, (y[inside], x[inside]), values)
    return image


def refine_track(events, points, step_us=1000, max_seed_gap_us=15000):
    """Refine supported circles; unsupported positions remain explicit seed fallbacks."""
    if step_us <= 0 or max_seed_gap_us <= 0:
        raise ValueError('time intervals must be positive')
    points = np.asarray(points, dtype=float).reshape(-1, 4)
    t, keep, curves, radius = robust_seed(points)
    good = points[keep]
    ts = np.arange(int(good[0, 0]), int(good[-1, 0])+1, step_us, dtype=np.int64)
    ordered = events[np.argsort(events['timestamp'], kind='stable')]
    event_t = ordered['timestamp'].astype(np.int64, copy=False)
    records = []
    for now in ts:
        right = np.searchsorted(good[:, 0], now, side='right')
        if right and right < len(good) and good[right, 0]-good[right-1, 0] > max_seed_gap_us:
            continue
        time_ms = (now - points[0, 0])/1000.0
        seed = np.array([curve(time_ms) for curve in curves], dtype=float)
        velocity = np.array([curve.derivative()(time_ms)/1000 for curve in curves], dtype=float)
        r = float(np.interp(now, good[:, 0], radius))
        half = max(32, int(np.ceil(r+18)))
        lo, hi = np.searchsorted(event_t, now-1000, side='left'), np.searchsorted(event_t, now, side='left')
        image = warped_local_image(ordered[lo:hi], int(now), seed, velocity, half)
        detector = HoughCircleBallDetector({'predictive_search_px': 8, 'predictive_search_step_px': 2,
                    'predictive_radius_search_px': 4, 'predictive_radius_step_px': 2,
                    'predictive_refine_candidates': True, 'predictive_nms_px': 3})
        found = detector._search_ring_candidates_near(image, half, half, r, 5)
        ranked = sorted(found, key=lambda c: c[3] - .008*np.hypot(c[0]-half, c[1]-half) - .008*abs(c[2]-r), reverse=True)
        x, y, rr, score = ranked[0]
        supported = bool(score >= .22 and np.hypot(x-half, y-half) <= 9 and abs(rr-r) <= 5)
        center = seed + np.array([x-half, y-half]) if supported else seed
        records.append([float(now), *center, float(rr if supported else r), float(score), float(supported), *seed])
    out = np.asarray(records, dtype=float).reshape(-1, 8)
    # Smooth only runs of supported measurements; never smooth across missing support.
    valid = out[:, 5] > 0
    indices = np.flatnonzero(valid)
    breaks = (np.diff(indices) != 1) | (np.diff(out[indices, 0]) != step_us)
    runs = np.split(indices, np.flatnonzero(breaks)+1)
    for run in runs:
        if len(run) >= 5:
            residual = out[run, 1:3] - out[run, 6:8]
            out[run, 1:3] = out[run, 6:8] + savgol_filter(residual, 5, 2, axis=0)
            out[run, 3] = savgol_filter(out[run, 3], 5, 2)
        if len(run) >= 9:
            residual = out[run, 1:3] - out[run, 6:8]
            out[run, 1:3] = out[run, 6:8] + savgol_filter(residual, 9, 2, axis=0)
            out[run, 3] = savgol_filter(out[run, 3], 9, 2)
    return out, keep


def refine_track_adaptive(events, points, step_us=1000, max_seed_gap_us=15000):
    """Experimental wider radius search, with radius continuity and explicit fallbacks.

    This tracks within the time span covered by coarse detections; it is not
    full-frame reacquisition, nor is circular event support a ground truth.
    """
    if step_us <= 0 or max_seed_gap_us <= 0:
        raise ValueError('time intervals must be positive')
    points = np.asarray(points, dtype=float).reshape(-1, 4)
    _, keep, curves, radii = robust_seed(points)
    good = points[keep]
    ordered = events[np.argsort(events['timestamp'], kind='stable')]
    event_t = ordered['timestamp'].astype(np.int64, copy=False)
    records = []
    previous_radius = None
    previous_time = None
    for now in np.arange(int(good[0, 0]), int(good[-1, 0])+1, step_us, dtype=np.int64):
        right = np.searchsorted(good[:, 0], now, side='right')
        if right and right < len(good) and good[right, 0]-good[right-1, 0] > max_seed_gap_us:
            previous_radius = previous_time = None
            continue
        ms = (now-points[0, 0])/1000.
        seed = np.array([c(ms) for c in curves])
        velocity = np.array([c.derivative()(ms)/1000 for c in curves])
        base = float(np.interp(now, good[:, 0], radii))
        if previous_time is None or now-previous_time > 5000:
            prior = base
        else:
            prior = .8*previous_radius + .2*base
        search = max(8, min(45, int(np.ceil(base*.45))))
        half = max(40, int(np.ceil(prior+search+16)))
        lo, hi = np.searchsorted(event_t, now-1000), np.searchsorted(event_t, now)
        image = warped_local_image(ordered[lo:hi], int(now), seed, velocity, half)
        detector = HoughCircleBallDetector({'predictive_search_px': 8, 'predictive_search_step_px': 4,
            'predictive_radius_search_px': search, 'predictive_radius_step_px': 3,
            'predictive_refine_candidates': False, 'predictive_nms_px': 4})
        found = detector._search_ring_candidates_near(image, half, half, prior, 12)
        if previous_time is not None and now-previous_time <= 5000:
            max_change = max(2., 2.5*(now-previous_time)/1000.)
            found = [c for c in found if abs(c[2]-previous_radius) <= max_change]
        if not found:
            records.append([float(now), *seed, float(prior), 0., 0., *seed])
            continue
        ranked = sorted(found, key=lambda c: c[3] - .006*np.hypot(c[0]-half, c[1]-half)
                        - .0015*abs(c[2]-prior), reverse=True)
        x, y, rr, score = ranked[0]
        supported = bool(score >= .22 and np.hypot(x-half, y-half) <= 10 and rr >= 8)
        if supported:
            previous_radius, previous_time = rr, int(now)
            center = seed + np.array([x-half, y-half])
        else:
            center, rr = seed, prior
        records.append([float(now), *center, float(rr), float(score), float(supported), *seed])
    out = np.asarray(records, float).reshape(-1, 8)
    indices = np.flatnonzero(out[:, 5] > 0)
    breaks = (np.diff(indices) != 1) | (np.diff(out[indices, 0]) != step_us)
    for run in np.split(indices, np.flatnonzero(breaks)+1):
        if len(run) >= 5:
            residual = out[run, 1:3]-out[run, 6:8]
            out[run, 1:3] = out[run, 6:8]+savgol_filter(residual, 5, 2, axis=0)
            out[run, 3] = savgol_filter(out[run, 3], 5, 2)
        if len(run) >= 9:
            residual = out[run, 1:3]-out[run, 6:8]
            out[run, 1:3] = out[run, 6:8]+savgol_filter(residual, 9, 2, axis=0)
            out[run, 3] = savgol_filter(out[run, 3], 9, 2)
    return out, keep
