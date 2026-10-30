import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


tools_dir = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(tools_dir))
spec = importlib.util.spec_from_file_location("live_measure", tools_dir / "live_measure.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_activity_peak_windows_require_both_cameras_and_separate_flights():
    edges = np.arange(0, 1_020_000, 20_000, dtype=np.int64)
    master = np.full(50, 10); slave = np.full(50, 10)
    master[[5, 6, 30, 40]] = [5000, 4000, 6000, 9000]
    slave[[5, 6, 30, 40]] = [5000, 4500, 7000, 1]
    windows = module.select_activity_peaks(
        edges, master, slave, 0, maximum_windows=4,
        separation_us=200_000, pre_peak_us=40_000, post_peak_us=300_000)
    assert len(windows) == 2
    assert windows[0]["start_offset_s"] == pytest.approx(.07)
    assert windows[1]["start_offset_s"] == pytest.approx(.57)
    assert windows[1]["duration_s"] == pytest.approx(.34)
    assert all(window["source"] == "raw_stereo_activity" for window in windows)


class FakeTool:
    def __init__(self):
        self.values = {}

    def set_param(self, name, value):
        self.values[name] = value
        return True

    def get_param(self, name):
        return True, self.values[name]


class FakeCamera:
    def __init__(self):
        self.tools = {}
        self.batch_us = None

    def get_tool(self, tool_type):
        key = int(tool_type)
        self.tools.setdefault(key, FakeTool())
        return self.tools[key]

    def set_batch_events_time(self, batch_us):
        self.batch_us = batch_us


def test_camera_configuration_returns_checked_hardware_readback():
    camera = FakeCamera()
    readback = module.configure_camera(
        camera, "MASTER", 2000, 10, False, 0, 0, False, 1)
    assert camera.batch_us == 2000
    assert readback == {
        "sync_mode": "MASTER",
        "batch_us": 2000,
        "event_rate_control": {"enabled": False, "maximum_rate_mevs": None},
        "anti_flicker_enabled": False,
        "bias_diff_on": 0,
        "bias_diff_off": 0,
        "trail_filter": {"enabled": False, "threshold_ms": None},
    }


def test_result_row_keeps_spin_empty_without_validated_channel_output():
    class Result:
        timestamp_us = 123
        master_track = slave_track = None
        measured_position_m = filtered_position_m = velocity_mps = None
        reprojection_error_px = None
        speed_mps = None

    row = module.result_row(Result(), None)
    assert row["timestamp_us"] == 123
    assert row["spin_rpm"] == ""
    assert row["omega_x_rad_s"] == ""
