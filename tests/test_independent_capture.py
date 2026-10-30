"""Exercise dual-capture worker lifecycle without attached hardware."""
import queue
import sys
import types
from enum import IntEnum
from threading import Event

import numpy as np

from tools.dual_capture_isolated import camera_worker


class FakeType(IntEnum):
    BIAS = 0
    SYNC = 2
    EVENT_RATE_CONTROL = 3
    EVENT_TRAIL_FILTER = 4
    ANTI_FLICKER = 5
    ROI = 6


class FakeTool:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def set_param(self, name, value):
        self.values[name] = value
        return True

    def get_param(self, name):
        return (name in self.values, self.values.get(name))


class FakeCamera:
    def __init__(self, start_result=0):
        self.tools = {kind: FakeTool() for kind in FakeType}
        self.tools[FakeType.ROI] = FakeTool({'x': 0, 'y': 0, 'width': 4, 'height': 3})
        self.started_result = start_result
        self.recorded = []
        self.record_stopped = False
        self.camera_stopped = False
        self.ts = 0

    def get_tool(self, kind): return self.tools[kind]
    def get_width(self): return 4
    def get_height(self): return 3
    def set_batch_events_time(self, value): self.batch = value
    def start(self): return self.started_result
    def stop(self): self.camera_stopped = True; return 0
    def start_recording(self, path): self.recorded.append(path); return 0
    def stop_recording(self): self.record_stopped = True; return 0
    def get_next_batch(self):
        self.ts += 1000
        return np.array([(1, 1, 1, self.ts)], dtype=[('x', 'u2'), ('y', 'u2'), ('polarity', 'u1'), ('timestamp', 'u8')])


def install_fake(monkeypatch, camera):
    manager = types.ModuleType('dvsense_driver.camera_manager')
    class FakeManager:
        def update_cameras(self): pass
        def open_camera(self, serial): return camera
    manager.DvsCameraManager = FakeManager
    hal = types.ModuleType('dvsense_driver.hal.camera_tool')
    hal.ToolType = FakeType
    monkeypatch.setitem(sys.modules, 'dvsense_driver.camera_manager', manager)
    monkeypatch.setitem(sys.modules, 'dvsense_driver.hal.camera_tool', hal)


def test_worker_records_and_stops(monkeypatch, tmp_path):
    camera = FakeCamera()
    install_fake(monkeypatch, camera)
    gate, started = Event(), Event(); gate.set()
    commands, messages, preview = queue.Queue(), queue.Queue(), queue.Queue(maxsize=1)
    commands.put({'action': 'start', 'directory': str(tmp_path)})
    commands.put({'action': 'stop'})
    commands.put({'action': 'quit'})
    camera_worker('master', 'serial-F3E', 1000, gate, started, commands, messages, preview, Event())
    events = list(messages.queue)
    assert started.is_set()
    assert [item[0] for item in events] == ['configured', 'started', 'recording', 'recorded', 'statistics', 'closed']
    assert camera.recorded == [str(tmp_path / 'master_serial-F3E.raw')]
    assert camera.record_stopped and camera.camera_stopped
    config = events[0][2]
    assert all(config[name] == 0 for name in ('bias_fo', 'bias_hpf', 'bias_diff_on', 'bias_diff', 'bias_diff_off', 'bias_refr'))
    assert config['sync_mode'] == 'MASTER'


def test_worker_fails_closed_when_camera_start_fails(monkeypatch):
    camera = FakeCamera(start_result=7)
    install_fake(monkeypatch, camera)
    gate, started = Event(), Event(); gate.set()
    messages = queue.Queue()
    camera_worker('slave', 'serial-F59', 1000, gate, started, queue.Queue(), messages, queue.Queue(), Event())
    events = list(messages.queue)
    assert not started.is_set()
    assert any(kind == 'error' and '返回码 7' in details for kind, _, details in events)
    assert not camera.recorded


def test_cancel_before_start_does_not_start_camera(monkeypatch):
    camera = FakeCamera()
    install_fake(monkeypatch, camera)
    gate, started, shutdown = Event(), Event(), Event()
    gate.set(); shutdown.set()
    messages = queue.Queue()
    camera_worker('slave', 'serial-F59', 1000, gate, started, queue.Queue(), messages, queue.Queue(), shutdown)
    assert not started.is_set()
    assert not camera.recorded
    assert not camera.camera_stopped
