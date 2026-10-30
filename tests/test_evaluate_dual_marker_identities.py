import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "evaluate_dual_marker_identities.py"
SPEC = importlib.util.spec_from_file_location("evaluate_dual_marker_identities", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _payload(frames=20, markers=4):
    return {"frames": [{
        "timestamp_us": 1000 * index,
        "status": "reviewed",
        "master_labels": [{"id": marker, "x": marker * 20.0, "y": 50.0}
                          for marker in range(markers)],
        "slave_labels": [{"id": marker, "x": marker * 20.0 + 2, "y": 70.0}
                         for marker in range(markers)],
    } for index in range(frames)]}


def _candidates(payload, side, switch=False):
    result = {}
    key = f"{side}_labels"
    for frame_index, frame in enumerate(payload["frames"]):
        result[frame["timestamp_us"]] = [{
            "track_id": item["id"] + (100 if switch and frame_index >= 10 else 0),
            "x": item["x"], "y": item["y"],
        } for item in frame[key]]
    return result


def test_dual_identity_gate_accepts_stable_tracks_in_both_cameras():
    payload = _payload()
    report = MODULE.evaluate(payload, _candidates(payload, "master"),
                             _candidates(payload, "slave"))
    assert report["accepted"] is True
    assert report["spin_solver_allowed"] is True
    assert report["shared_stereo_identity_observations"] == 80
    assert report["master"]["identity_switch_fraction_of_matched_pairs"] == 0.0


def test_dual_identity_gate_rejects_switches_and_missing_review():
    payload = _payload()
    switched = MODULE.evaluate(payload, _candidates(payload, "master", switch=True),
                               _candidates(payload, "slave"))
    assert switched["accepted"] is False
    payload["frames"][0]["status"] = "unreviewed"
    incomplete = MODULE.evaluate(payload, _candidates(payload, "master"),
                                 _candidates(payload, "slave"))
    assert incomplete["accepted"] is False
