import pytest

from pingpong_event_pipeline.acquisition import (
    configure_boolean_parameter, configure_event_rate_control,
    configure_trail_filter, spin_capture_config_valid)


class FakeTool:
    def __init__(self, fail=None, readback=None):
        self.values = {}
        self.fail = fail
        self.readback = readback or {}

    def set_param(self, name, value):
        if name == self.fail:
            return False
        self.values[name] = value
        return True

    def get_param(self, name):
        return True, self.readback.get(name, self.values.get(name))


def test_erc_is_disabled_without_configuring_a_rate():
    tool = FakeTool()
    result = configure_event_rate_control(tool, False, 10, "MASTER")
    assert tool.values == {"enable": False}
    assert result == {"enabled": False, "maximum_rate_mevs": None}


def test_enabled_erc_sets_and_reads_back_rate():
    tool = FakeTool()
    result = configure_event_rate_control(tool, True, 12, "SLAVE")
    assert tool.values == {"max_event_rate": 12, "enable": True}
    assert result == {"enabled": True, "maximum_rate_mevs": 12}


def test_erc_readback_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="回读失败"):
        configure_event_rate_control(
            FakeTool(readback={"enable": True}), False, 10, "MASTER")


def test_trail_filter_is_disabled_and_read_back():
    tool = FakeTool()
    result = configure_trail_filter(tool, False, 1, "MASTER")
    assert tool.values == {"enable": False}
    assert result == {"enabled": False, "threshold_ms": None}


def test_enabled_trail_filter_sets_threshold_and_type():
    tool = FakeTool()
    result = configure_trail_filter(tool, True, 2, "SLAVE")
    assert tool.values == {"threshold": 2, "type": "TRAIL", "enable": True}
    assert result == {"enabled": True, "threshold_ms": 2}


def test_trail_filter_readback_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="回读失败"):
        configure_trail_filter(
            FakeTool(readback={"enable": True}), False, 1, "MASTER")


def test_spin_capture_config_requires_known_filters_off():
    valid = {"bias_diff_on": 0, "bias_diff_off": 0,
             "trail_filter_enabled": False,
             "event_rate_control_enabled": False}
    assert spin_capture_config_valid(valid) is True
    assert spin_capture_config_valid({**valid, "event_rate_control_enabled": True}) is False
    assert spin_capture_config_valid({"trail_filter_enabled": False}) is False


def test_boolean_parameter_is_checked():
    tool = FakeTool()
    assert configure_boolean_parameter(tool, "enable", False, "Anti Flicker") is False
    assert tool.values["enable"] is False
    with pytest.raises(RuntimeError, match="回读失败"):
        configure_boolean_parameter(
            FakeTool(readback={"enable": True}), "enable", False, "Anti Flicker")
