import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from render_marker_evidence import assign_persistent, evidence_image


def test_consecutive_gate_does_not_extrapolate():
    candidate = [{'x': 10, 'y': 10}]
    first = assign_persistent([], candidate, min_streak=2)
    assert not first[0]['show']
    second = assign_persistent(first, candidate, min_streak=2)
    assert second[0]['show']
    assert assign_persistent(second, [], min_streak=2) == []
    assert not assign_persistent([], candidate, min_streak=2)[0]['show']


def test_evidence_blank_without_observed_support():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    evidence, mask, fits = evidence_image(image, [{'x': 128, 'y': 128}])
    assert not np.any(evidence) and not np.any(mask) and not fits
