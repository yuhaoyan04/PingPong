import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "finalize_physical_template.py"
SPEC = importlib.util.spec_from_file_location("finalize_physical_template", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _frame(status, shared):
    labels = [{"id": index + 1} for index in range(shared)]
    return {"status": status, "master_labels": labels, "slave_labels": labels}


def test_annotation_readiness_requires_review_count_observations_and_connectivity():
    incomplete = MODULE.annotation_readiness({
        "frames": [_frame("reviewed", 3) for _ in range(19)]})
    assert incomplete["ready"] is False
    assert incomplete["checks"]["reviewed_frames"] is False
    assert incomplete["checks"]["shared_identity_observations"] is False

    ready = MODULE.annotation_readiness({
        "frames": [_frame("reviewed", 3) for _ in range(20)]})
    assert ready["ready"] is True
    assert ready["shared_identity_observations"] == 60
    assert ready["frames_with_at_least_3_shared_ids"] == 20
