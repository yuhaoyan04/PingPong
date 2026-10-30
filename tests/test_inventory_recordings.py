import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "inventory_recordings", Path(__file__).parents[1] / "tools" / "inventory_recordings.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_old_capture_metadata_is_unknown_and_not_spin_valid():
    audit = module.acquisition_audit({"max_event_rate_mevs": 10})
    assert audit["known"] is False
    assert audit["spin_capture_config_valid"] is False


def test_unfiltered_unlimited_capture_is_spin_configuration_valid():
    audit = module.acquisition_audit({
        "bias_diff_on": 0, "bias_diff_off": 0,
        "trail_filter_enabled": False,
        "event_rate_control_enabled": False,
    })
    assert audit["known"] is True
    assert audit["spin_capture_config_valid"] is True


def test_event_dropping_configuration_is_rejected_for_spin():
    audit = module.acquisition_audit({"acquisition_config": {
        "bias_diff_on": 0, "bias_diff_off": 0,
        "trail_filter_enabled": False,
        "event_rate_control_enabled": True,
    }})
    assert audit["known"] is True
    assert audit["spin_capture_config_valid"] is False
