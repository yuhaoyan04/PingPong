import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "offline_pooled_spin", Path(__file__).parents[1] / "tools" / "offline_pooled_spin.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_multibaseline_threshold_schedule_matches_validated_settings():
    assert [module.angular_threshold_for_lag(.01, lag, True)
            for lag in (1, 3, 5, 7)] == [.01, .015, .02, .025]


def test_threshold_scaling_can_be_disabled_for_compatible_single_runs():
    assert module.angular_threshold_for_lag(.01, 7, False) == .01
