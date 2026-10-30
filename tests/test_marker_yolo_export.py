import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from export_marker_yolo import rectangle_to_yolo, main


def test_box_conversion():
    shape = {'label': 'marker', 'shape_type': 'rectangle', 'points': [[20, 30], [10, 10]]}
    assert rectangle_to_yolo(shape, 100, 100) == '0 0.15000000 0.20000000 0.10000000 0.20000000'


def test_invalid_or_unknown_annotation_rejected():
    with pytest.raises(ValueError):
        rectangle_to_yolo({'label': 'unknown', 'shape_type': 'rectangle', 'points': [[0, 0], [2, 2]]}, 100, 100)
    with pytest.raises(ValueError):
        rectangle_to_yolo({'label': 'marker', 'shape_type': 'rectangle', 'points': [[-1, 0], [2, 2]]}, 100, 100)


def test_export_only_explicitly_annotated_groups(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    for group in ('serve1', 'serve2'):
        folder = dataset / group / 'binary'
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / 'sample.png'), np.zeros((20, 20), np.uint8))
        cv2.imwrite(str(folder / 'unreviewed.png'), np.zeros((20, 20), np.uint8))
        (folder / 'sample.json').write_text(json.dumps({'imagePath': 'sample.png', 'imageWidth': 20, 'flags': {'reviewed': True},
            'imageHeight': 20, 'shapes': [{'label': 'marker', 'shape_type': 'rectangle',
                                           'points': [[2, 4], [10, 12]]}]}), encoding='utf-8')
    out = tmp_path / 'export'
    monkeypatch.setattr(sys, 'argv', ['export', '--dataset', str(dataset), '--output', str(out),
                                      '--train', 'serve1', '--val', 'serve2'])
    main()
    assert len(list((out / 'images').rglob('*.png'))) == 2
    assert 'channels: 1' in (out / 'dataset.yaml').read_text()
    assert len(list((out / 'labels').rglob('*.txt'))) == 2


def test_train_only_uses_both_groups_without_claiming_validation(tmp_path, monkeypatch):
    dataset = tmp_path/'data'
    for group in ('serve01_F3E','serve06_F3E'):
        folder=dataset/group/'binary'; folder.mkdir(parents=True)
        cv2.imwrite(str(folder/'a.png'),np.zeros((20,20),np.uint8))
        (folder/'a.json').write_text(json.dumps({'imagePath':'a.png','imageWidth':20,'imageHeight':20,
              'flags':{'reviewed':True},'shapes':[]}),encoding='utf8')
    out=tmp_path/'export'
    monkeypatch.setattr(sys,'argv',['export','--dataset',str(dataset),'--output',str(out),'--train',
                                  'serve01_F3E','serve06_F3E','--train-only'])
    main()
    assert len(list((out/'images'/'train').glob('*.png'))) == 2
    assert not (out/'images'/'val').exists()
    assert 'val: images/train' in (out/'dataset.yaml').read_text()
    assert json.loads((out/'split_manifest.json').read_text())['train_only'] is True
