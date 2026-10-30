import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from manual_calibration_capture import detect_grid, save_pair
from live_stereo_view import make_blob_detector


def test_blank_image_has_no_complete_grid():
    assert detect_grid(np.zeros((720, 1280), np.uint8), make_blob_detector()) is None


def test_save_pair_keeps_native_images_and_timestamps(tmp_path):
    frames = {'master': np.full((12, 20), 42, np.uint8),
              'slave': np.full((12, 20), 99, np.uint8)}
    meta = {'master': {'first_event_us': 10, 'last_event_us': 20},
            'slave': {'first_event_us': 11, 'last_event_us': 21}}
    saved = save_pair(tmp_path, 1, frames, meta)
    assert saved['preview_windows'] == meta
    assert cv2.imread(str(tmp_path / 'master_001.png'), 0).shape == (12, 20)
    assert int(cv2.imread(str(tmp_path / 'slave_001.png'), 0)[0, 0]) == 99
    assert json.loads((tmp_path / 'pair_001.json').read_text())['index'] == 1
    with pytest.raises(RuntimeError):
        save_pair(tmp_path, 1, frames, meta)
