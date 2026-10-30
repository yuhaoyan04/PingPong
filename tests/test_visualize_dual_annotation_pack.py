import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "visualize_dual_annotation_pack.py"
SPEC = importlib.util.spec_from_file_location("visualize_dual_annotation_pack", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_align_records_pairs_exact_timestamps_and_reports_missing():
    left = {1000: {"side": "l0"}, 2000: {"side": "l1"}, 3000: {"side": "l2"}}
    right = {1000: {"side": "r0"}, 3000: {"side": "r2"}, 4000: {"side": "r3"}}

    pairs, left_only, right_only = MODULE.align_records(left, right)

    assert [item[0] for item in pairs] == [1000, 3000]
    assert pairs[1][1]["side"] == "l2"
    assert pairs[1][2]["side"] == "r2"
    assert left_only == [2000]
    assert right_only == [4000]


def test_distribution_is_explicit_for_empty_and_nonempty_inputs():
    assert MODULE.distribution([]) == {
        "median": 0.0, "p10": 0.0, "p90": 0.0, "minimum": 0, "maximum": 0}
    result = MODULE.distribution([1, 2, 9])
    assert result["median"] == 2.0
    assert result["minimum"] == 1
    assert result["maximum"] == 9
