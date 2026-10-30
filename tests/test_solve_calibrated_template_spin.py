import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "tools" / "solve_calibrated_template_spin.py"
SPEC = importlib.util.spec_from_file_location("solve_calibrated_template_spin", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_time_range_overlap_is_inclusive_and_handles_missing_metadata():
    assert MODULE.time_ranges_overlap(10, 20, 20, 30) is True
    assert MODULE.time_ranges_overlap(10, 19, 20, 30) is False
    assert MODULE.time_ranges_overlap(None, None, 20, 30) is False


def test_rejected_report_never_contains_candidate_spin(tmp_path):
    report = MODULE.rejected_report("gate failed", tmp_path / "template.json",
                                    tmp_path / "markers.npz")
    assert report["spin_solver_allowed"] is False
    assert report["spin_rpm"] is None
    assert report["omega_rad_s_camera"] is None


def test_cli_accepts_valid_independent_synthetic_template_sequence(tmp_path, monkeypatch):
    rng = np.random.default_rng(950)
    template = rng.normal(size=(12, 3))
    template /= np.linalg.norm(template, axis=1, keepdims=True)
    template_path = tmp_path / "template.npz"
    np.savez_compressed(template_path, directions=template)
    report_path = tmp_path / "template.json"
    report_path.write_text(json.dumps({
        "physical_template_geometry_accepted": True,
        "candidate_tracking_identity_accepted": True,
        "spin_solver_allowed": True,
        "calibration_timestamp_start_us": 0,
        "calibration_timestamp_end_us": 20000,
    }), encoding="utf-8")
    timestamps = 100000 + np.arange(18, dtype=np.int64) * 1000
    omega = np.array([80.0, -40.0, 60.0])
    directions = []
    offsets = [0]
    for timestamp in timestamps:
        rotation = cv2.Rodrigues(omega * ((timestamp - timestamps[0]) * 1e-6))[0]
        observed = (rotation @ template.T).T
        directions.append(observed); offsets.append(offsets[-1] + len(observed))
    markers_path = tmp_path / "markers.npz"
    np.savez_compressed(markers_path, timestamps_us=timestamps,
                        offsets=np.asarray(offsets), directions=np.vstack(directions))
    output = tmp_path / "spin.json"
    monkeypatch.setattr(sys, "argv", [
        "solve_calibrated_template_spin.py",
        "--template", str(template_path),
        "--template-report", str(report_path),
        "--marker-sets", str(markers_path),
        "--output", str(output),
    ])
    MODULE.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    expected_rpm = np.linalg.norm(omega) * 60 / (2 * np.pi)
    assert result["status"] == "accepted"
    assert result["calibration_measurement_overlap"] is False
    assert abs(result["spin_rpm"] - expected_rpm) < 1.0
    assert result["local_track_frames"] >= 15
