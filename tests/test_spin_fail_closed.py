import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_pooled_spin_summary_never_promotes_unvalidated_candidate(tmp_path):
    timestamps = np.arange(10, dtype=np.int64) * 1000
    points = np.column_stack((timestamps * 1e-6, np.zeros(10), np.ones(10)))
    stereo_path = tmp_path / "stereo_path.npz"
    np.savez(stereo_path, timestamps_us=timestamps, points_m=points,
             detection_step_us=1000)
    spin_paths = []
    for index, rpm in enumerate((99.0, 100.0, 101.0)):
        path = tmp_path / f"spin_{index}.json"
        path.write_text(json.dumps({
            "rpm": rpm,
            "axis_master": [1.0, 0.0, 0.0],
            "camera_axis_angle_deg": 1.0,
            "camera_relative_rpm_difference": .01,
            "inliers": 600,
            "condition": 2.0,
        }), encoding="utf-8")
        spin_paths.append(path)
    output = tmp_path / "measurement.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "tools" / "summarize_pooled_spin.py"),
        "--spin-json", *map(str, spin_paths),
        "--stereo-path", str(stereo_path),
        "--first-index", "0", "--last-index", "9",
        "--output", str(output),
    ], cwd=root, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["internal_consistency_only"] is True
    assert report["candidate_spin_rpm_unvalidated"] == 100.0
    assert report["spin_rpm"] is None
    assert report["status"] == "spin_unvalidated"
    assert report["gates"]["physical_marker_identity_and_absolute_spin_validation"] is False
