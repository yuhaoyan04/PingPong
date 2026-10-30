import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from marker_annotation_player import Window


def test_edit_autosaves_and_empty_is_not_implicitly_negative(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = tmp_path / 'serve01_F3E'
    (group / 'binary').mkdir(parents=True)
    (group / 'polarity_reference').mkdir()
    cv2.imwrite(str(group / 'binary' / 'frame.png'), np.zeros((256, 256), np.uint8))
    cv2.imwrite(str(group / 'polarity_reference' / 'frame.png'), np.zeros((256, 256, 3), np.uint8))
    (group / 'source.json').write_text(json.dumps({'camera': 'F3E'}), encoding='utf-8')
    row = {'image': 'binary/frame.png', 'polarity_reference': 'polarity_reference/frame.png',
           'timestamp_us': '1000', 'center_x_px': '100', 'center_y_px': '100', 'radius_px': '30'}
    with (group / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)

    class Video:
        def isOpened(self): return True
        def get(self, prop): return 1
        def set(self, prop, value): return True
        def read(self): return True, np.zeros((720, 2560, 3), np.uint8)
        def release(self): pass

    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: Video())
    window = Window(group, tmp_path / 'video.mp4', 1000, 5000)
    window.canvas.boxes.append([10, 20, 30, 40])
    window.save()
    saved = group / 'binary' / 'frame.json'
    assert json.loads(saved.read_text(encoding='utf-8'))['flags']['reviewed'] is True
    window.undo()
    data = json.loads(saved.read_text(encoding='utf-8'))
    assert data['shapes'] == [] and data['flags']['reviewed'] is False
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *args: QtWidgets.QMessageBox.Yes)
    window.negative()
    data = json.loads(saved.read_text(encoding='utf-8'))
    assert data['shapes'] == [] and data['flags']['reviewed'] is True
    window.close()
