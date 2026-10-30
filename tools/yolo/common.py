import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Paths may be overridden for a new, isolated training run.
ROOT = Path(os.environ.get('PINGPONG_ROOT', ROOT)).resolve()
DATA = Path(os.environ.get('PINGPONG_DATASET', ROOT / 'data/server_marker_two_camera_train_v1'))
EXP = Path(os.environ.get('PINGPONG_EXPERIMENT', ROOT / 'experiments/two_camera_v1'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(EXP / 'ultralytics_config'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def table(path, records, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        w.writeheader()
        w.writerows(records)
