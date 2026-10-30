import json

import numpy as np
import pytest

from pingpong_event_pipeline.template_spin_channel import load_validated_template


def test_load_validated_template_requires_every_gate(tmp_path):
    template = tmp_path / "template.npz"
    np.savez_compressed(template, directions=np.eye(3))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "physical_template_geometry_accepted": True,
        "candidate_tracking_identity_accepted": False,
        "spin_solver_allowed": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_tracking_identity_accepted"):
        load_validated_template(template, report)


def test_load_validated_template_normalizes_valid_directions(tmp_path):
    template = tmp_path / "template.npz"
    directions = np.array([[2., 0, 0], [0, 3, 0], [0, 0, 4], [-2, 0, 0]])
    np.savez_compressed(template, directions=directions)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "physical_template_geometry_accepted": True,
        "candidate_tracking_identity_accepted": True,
        "spin_solver_allowed": True,
    }), encoding="utf-8")
    loaded, _ = load_validated_template(template, report)
    assert np.allclose(np.linalg.norm(loaded, axis=1), 1.0)
