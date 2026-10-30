import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "evaluate_marker_identity",
    Path(__file__).parents[1] / "tools" / "evaluate_marker_identity.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def frame(index, labels, status="reviewed"):
    return {"local_path_index": index, "identity_labels": [
        {"id": identifier, "x": x, "y": y} for identifier, x, y in labels
    ], "status": status}


def candidate(track_id, x, y):
    return {"track_id": track_id, "x": x, "y": y}


def test_identity_evaluation_accepts_sufficient_perfect_sequence():
    payload = {"frames": [frame(i, [(1, i, 0), (2, i, 20)]) for i in range(30)]}
    candidates = {i: [candidate(11, i + .5, 0), candidate(12, i, 20.5)]
                  for i in range(30)}
    report, rows = module.evaluate_identity(payload, candidates, 2)
    assert report["physical_label_observations"] == 60
    assert report["consecutive_physical_identity_pairs"] == 58
    assert report["framewise_label_recall"] == 1
    assert report["identity_retention_fraction_all_pairs"] == 1
    assert report["identity_switch_fraction_of_matched_pairs"] == 0
    assert report["accepted"] is True
    assert {row["result"] for row in rows} == {"retained"}


def test_identity_evaluation_counts_switch_and_unmatched_pair():
    payload = {"frames": [
        frame(0, [(1, 0, 0)]), frame(1, [(1, 1, 0)]), frame(2, [(1, 2, 0)])]}
    candidates = {
        0: [candidate(5, 0, 0)],
        1: [candidate(6, 1, 0)],
        2: [candidate(6, 100, 100)],
    }
    report, rows = module.evaluate_identity(payload, candidates, 3)
    assert report["switched_candidate_id_pairs"] == 1
    assert report["unmatched_identity_pairs"] == 1
    assert report["identity_switch_fraction_of_matched_pairs"] == 1
    assert report["accepted"] is False
    assert [row["result"] for row in rows] == ["switched", "unmatched"]
