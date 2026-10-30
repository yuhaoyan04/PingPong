import json
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from marker_dataset_launcher import create_window


def test_switches_between_original_and_refined_datasets(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for dataset, group in [('marker_annotation_all_v1', 'serve01_F3E'),
                           ('marker_annotation_refined_v2', 'serve06_F3E'),
                           ('marker_annotation_event_v1', 'serve09_F3E'),
                           ('marker_annotation_camera_enhanced_v1', 'serve05_F59')]:
        path = tmp_path/'output'/dataset/group
        path.mkdir(parents=True)
        (path/'manifest.csv').write_text('frame_index\n0\n')
        (path/'source.json').write_text(json.dumps({'frames': 1}))
    window = create_window(tmp_path)
    choices = window.findChildren(QtWidgets.QComboBox)
    assert 'serve01_F3E' in choices[1].itemText(0)
    choices[0].setCurrentIndex(1)
    assert choices[1].count() == 1 and 'serve06_F3E' in choices[1].itemText(0)
    choices[0].setCurrentIndex(2)
    assert choices[1].count() == 1 and 'serve09_F3E' in choices[1].itemText(0)
    choices[0].setCurrentIndex(3)
    assert choices[1].count() == 1 and 'serve05_F59' in choices[1].itemText(0)
    window.close()
