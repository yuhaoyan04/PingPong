import numpy as np

from pingpong_event_pipeline.models import Detection2D
from tools.build_serve_review_clips import fit_global_rule_path


def test_global_tracker_rejects_static_distractor_and_extends_through_bend():
    times = np.arange(90, dtype=np.int64) * 1000
    x = 100.0 + 4.0 * np.arange(90)
    y = 80.0 + 0.018 * np.arange(90) ** 2
    # A late image-plane bend is not represented by the quadratic RANSAC core;
    # the anchored endpoint chain should retain it without jumping to clutter.
    y[65:] += 0.08 * (np.arange(65, 90) - 65) ** 2
    candidates = []
    rng = np.random.default_rng(8)
    for index, timestamp in enumerate(times):
        values = [Detection2D(int(timestamp), float(x[index] + rng.normal(0, .5)),
                              float(y[index] + rng.normal(0, .5)), 29.0,
                              .44, 400)]
        # A stronger stationary response models the net/table background.
        values.append(Detection2D(int(timestamp), 720.0, 330.0, 31.0, .52, 700))
        candidates.append(values)

    path = fit_global_rule_path(times, candidates, hypotheses=800, seed=4)

    assert path is not None
    assert len(path["timestamps"]) >= 84
    assert np.count_nonzero(path["scores"] > 0) >= 82
    rows = (path["timestamps"] // 1000).astype(int)
    error = np.linalg.norm(path["values"][:, :2] - np.column_stack((x[rows], y[rows])), axis=1)
    assert np.median(error) < 2.0
    assert np.max(error) < 9.0


def test_global_tracker_reacquires_after_bounce_and_short_blackout():
    times = np.arange(112, dtype=np.int64) * 1000
    index = np.arange(len(times), dtype=np.float64)
    x = 90.0 + 3.4 * index
    y = 92.0 + 2.7 * index
    impact = 55
    # The vertical velocity reverses at a table/net impact.  There is no ball
    # candidate for several bins around contact because collision events can
    # merge the ball into the table or net response.
    y[impact:] = y[impact - 1] - 3.2 * np.arange(1, len(times) - impact + 1)
    blackout = set(range(52, 59))

    candidates = []
    rng = np.random.default_rng(31)
    for frame, timestamp in enumerate(times):
        values = []
        if frame not in blackout:
            values.append(Detection2D(
                int(timestamp), float(x[frame] + rng.normal(0, .45)),
                float(y[frame] + rng.normal(0, .45)), 28.0, .46, 420))
        # Persistent high-score structure represents the table/net response.
        values.append(Detection2D(int(timestamp), 690.0, 335.0, 30.0, .58, 760))
        candidates.append(values)

    path = fit_global_rule_path(times, candidates, hypotheses=1200, seed=9)

    assert path is not None
    assert len(path["timestamps"]) >= 104
    assert np.count_nonzero(path["scores"] > 0) >= 98
    rows = (path["timestamps"] // 1000).astype(int)
    error = np.linalg.norm(
        path["values"][:, :2] - np.column_stack((x[rows], y[rows])), axis=1)
    assert np.median(error) < 2.0
    assert np.quantile(error, .95) < 8.0
    # The returned path must include both sides of the reversal.
    assert rows.min() <= 3
    assert rows.max() >= 108
