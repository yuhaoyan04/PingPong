"""Diagnostic flags for candidate ball tracks; not an accuracy certification."""
import numpy as np


def assess_candidate(records, coarse_count, step_us=1000):
    arr = np.asarray(records, dtype=float).reshape(-1, 8)
    if step_us <= 0 or not len(arr) or np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError('empty or unordered trajectory')
    supported = arr[:, 5] > 0
    runs = np.split(np.flatnonzero(supported),
                    np.where((np.diff(np.flatnonzero(supported)) != 1)
                             | (np.diff(arr[np.flatnonzero(supported), 0]) != step_us))[0]+1)
    longest = max((len(run) for run in runs), default=0)
    fraction = float(supported.mean())
    triple = supported[:-2] & supported[1:-1] & supported[2:]
    dt = np.diff(arr[:, 0])
    triple &= (dt[:-1] == step_us) & (dt[1:] == step_us)
    residual = np.linalg.norm(np.diff(arr[:, 1:3], n=2, axis=0), axis=1)[triple]
    p90 = float(np.percentile(residual, 90)) if len(residual) else None
    flags = []
    if coarse_count < 6: flags.append('few_coarse_detections')
    if fraction < .7: flags.append('low_event_support_fraction')
    if longest < 15: flags.append('short_contiguous_support')
    if p90 is None or p90 > 1.5: flags.append('large_second_difference_or_insufficient_samples')
    return {'status': 'unverified_candidate', 'sample_count': len(arr),
            'supported_samples': int(supported.sum()), 'support_fraction': fraction,
            'longest_supported_run_ms': longest * step_us / 1000,
            'second_difference_p90_px_per_step_squared': p90,
            'diagnostic_flags': flags,
            'warning': 'Flags are heuristics, not accuracy or truth; manual review required.'}
