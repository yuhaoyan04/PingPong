import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sae = load("estimate_spin_by_sae")
warp = load("estimate_spin_by_event_warp")


def test_event_warp_preserves_unit_sphere():
    points = np.array([[0., 0., 1.], [.3, -.4, np.sqrt(.75)]])
    result = warp.warp_points(points, np.array([.001, -.002]),
                              np.array([200., -100., 50.]))
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=1e-12)


def test_sae_robust_solver_recovers_shared_rotation():
    rng = np.random.default_rng(7)
    truth = np.array([230., -170., 90.])
    first_two = rng.normal(0., .004, (500, 2))
    last = (1. - first_two @ truth[:2]) / truth[2]
    equations = np.column_stack((first_two, last))
    equations += rng.normal(0., 2e-5, equations.shape)
    equations[:70] = rng.normal(0., .005, (70, 3))  # unrelated edge planes
    estimate, inliers, _ = sae.solve(equations, max_rpm=6000, seed=4,
                                     iterations=4000, threshold=.12)
    assert inliers.sum() > 350
    assert np.degrees(np.arccos(np.clip(
        estimate @ truth / np.linalg.norm(estimate) / np.linalg.norm(truth), -1, 1))) < 2
    assert abs(np.linalg.norm(estimate) / np.linalg.norm(truth) - 1) < .05
