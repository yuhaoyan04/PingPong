import numpy as np
import csv
import json
import subprocess
import sys
from pathlib import Path

from pingpong_event_pipeline.frames import TableFrame, spin_components_table


def test_table_frame_transforms_points_and_vectors(tmp_path):
    # Camera coordinates are table coordinates rotated 90 degrees about Z and translated.
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.2, -0.1, 1.0])
    path = tmp_path / "table.npz"
    np.savez(path, R_table_from_master=R, origin_master_m=origin)
    frame = TableFrame(path)
    assert np.allclose(frame.point(origin + np.array([1.0, 0.0, 0.0])), [0.0, -1.0, 0.0])
    assert np.allclose(frame.vector([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])


def test_spin_components_follow_forward_rolling_convention():
    rpm = 300.0
    omega = np.array([0.0, rpm * 2 * np.pi / 60.0, 0.0])
    result = spin_components_table(omega, [2.0, 0.0, 0.0])
    assert result["dominant_spin"] == "topspin"
    assert np.isclose(result["topspin_signed_rpm"], rpm)
    reverse = spin_components_table(-omega, [2.0, 0.0, 0.0])
    assert reverse["dominant_spin"] == "backspin"


def test_no_slip_roll_validator_accepts_known_rolling_truth(tmp_path):
    table = tmp_path / "table.npz"
    np.savez(table, R_table_from_master=np.eye(3), origin_master_m=np.zeros(3))
    states = tmp_path / "states.csv"
    rpm = 60.0 / (2.0 * np.pi * .020)
    fields = ["timestamp_us", "x_m", "y_m", "z_m", "spin_rpm",
              "spin_axis_x", "spin_axis_y", "spin_axis_z"]
    with states.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(40):
            writer.writerow({"timestamp_us": index * 10_000, "x_m": index * .01,
                             "y_m": 0, "z_m": .020, "spin_rpm": rpm,
                             "spin_axis_x": 0, "spin_axis_y": 1, "spin_axis_z": 0})
    report = tmp_path / "roll.json"
    tool = Path(__file__).parents[1] / "tools" / "validate_spin_roll.py"
    subprocess.run([sys.executable, str(tool), str(states), "--table-frame", str(table),
                    "--output", str(report)], check=True, capture_output=True, text=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "absolute_spin_valid"
    assert result["rolling_axis_signed_alignment_median"] > .99
