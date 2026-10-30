import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import cv2


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "annotate_dual_marker_identities.py"
SPEC = importlib.util.spec_from_file_location("annotate_dual_marker_identities", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_pack(path, timestamps):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "coordinate_system": "normalized_patch_pixels",
        "image_size_px": [256, 256],
        "frames": [{"timestamp_us": timestamp, "frame": f"f{timestamp}.png"}
                   for timestamp in timestamps],
    }), encoding="utf-8")


def test_build_payload_uses_only_exact_shared_timestamps(tmp_path):
    master = tmp_path / "m" / "annotations.json"
    slave = tmp_path / "s" / "annotations.json"
    _write_pack(master, [100, 200, 300])
    _write_pack(slave, [100, 300, 400])

    payload = MODULE.build_payload(master, slave)

    assert [frame["timestamp_us"] for frame in payload["frames"]] == [100, 300]
    assert payload["master_only_timestamps"] == [200]
    assert payload["slave_only_timestamps"] == [400]
    assert payload["frames"][0]["master_labels"] == []
    assert payload["frames"][0]["slave_labels"] == []


def test_shared_ids_and_next_identifier_span_both_cameras():
    frames = [{
        "master_labels": [{"id": 2}, {"id": 7}],
        "slave_labels": [{"id": 2}, {"id": 5}],
    }]
    assert MODULE.shared_ids(frames[0]) == {2}
    assert MODULE.next_identifier(frames) == 8


def test_patch_full_mapping_and_line_transform_preserve_incidence():
    record = {
        "center_x_full_image": 300.0,
        "center_y_full_image": 200.0,
        "radius_px_full_image": 64.0,
    }
    patch_point = np.array([160.0, 80.0])
    full_point = MODULE.patch_to_full(patch_point, record, 256)
    full_line = np.array([2.0, -3.0,
                          -(2.0 * full_point[0] - 3.0 * full_point[1])])
    patch_line = MODULE.full_line_to_patch(full_line, record, 256)
    assert MODULE.line_distance(patch_line, patch_point) < 1e-9


def test_optical_flow_propagation_preserves_ids_and_snaps_to_proposals():
    previous = np.zeros((128, 128, 3), np.uint8)
    current = np.zeros_like(previous)
    seeds = [{"id": 3, "x": 35.0, "y": 45.0},
             {"id": 8, "x": 82.0, "y": 70.0}]
    shift = np.array([6.0, -4.0])
    for item in seeds:
        old = np.rint([item["x"], item["y"]]).astype(int)
        new = np.rint(old + shift).astype(int)
        cv2.circle(previous, tuple(old), 5, (255, 255, 255), -1)
        cv2.circle(current, tuple(new), 5, (255, 255, 255), -1)
    proposals = [
        {"x": 41.0, "y": 41.0, "density": .9},
        {"x": 88.0, "y": 66.0, "density": .8},
        {"x": 110.0, "y": 100.0, "density": .2},
    ]
    propagated = MODULE.propagate_labels_optical_flow(
        previous, current, seeds, proposals)
    assert [item["id"] for item in propagated] == [3, 8]
    assert [(item["x"], item["y"]) for item in propagated] == [
        (41.0, 41.0), (88.0, 66.0)]
    assert all(item["source"] == "optical_flow_proposal" for item in propagated)
    assert all(item["confidence"] > .7 for item in propagated)
