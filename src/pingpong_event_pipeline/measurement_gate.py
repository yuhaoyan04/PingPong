"""Conservative reporting of image-space motion; no unvalidated metric output."""
from __future__ import annotations

import numpy as np


def image_motion_summary(timestamps_us, centers_px, radii_px, measured_scores, size=(1280, 720)):
    t = np.asarray(timestamps_us, dtype=np.int64)
    centers = np.asarray(centers_px, dtype=float)
    radii = np.asarray(radii_px, dtype=float)
    scores = np.asarray(measured_scores, dtype=float)
    if (len(t) < 3 or centers.shape != (len(t), 2) or radii.shape != (len(t),)
            or np.any(np.diff(t) <= 0) or not np.isfinite(centers).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError('invalid ball trajectory')
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('missing detection scores')
    observed_fraction = float(np.mean(scores > 0))
    x, y = centers[:, 0], centers[:, 1]
    edge_fraction = float(np.mean((x-radii < 0) | (x+radii >= size[0])
                                  | (y-radii < 0) | (y+radii >= size[1])))
    velocity = np.gradient(centers, (t - t[0]).astype(float) * 1e-6, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return {'duration_ms': float((t[-1]-t[0])/1000), 'output_samples': len(t),
            'measured_detection_fraction': observed_fraction, 'ball_outside_image_fraction': edge_fraction,
            'median_image_speed_px_per_s': float(np.median(speed)),
            'image_velocity_median_px_per_s': np.median(velocity, axis=0).tolist(),
            'metric_speed_m_per_s': None, 'spin_rpm': None, 'rotation_axis': None,
            'accepted': False,
            'warnings': ['Image-plane speed is not metric ball speed.',
                         'Trajectory has no independent pixel-ground-truth validation.',
                         'Metric calibration, timing, marker identity and spin truth are not validated.']}
