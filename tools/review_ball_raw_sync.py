"""Audit actual serve RAWs and render independently decoded equal-time views."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.raw_sync import extract_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--camera', choices=['F3E', 'F59'], required=True)
    p.add_argument('--dataset', type=Path, default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    args = p.parse_args()
    report = extract_camera(args.raw, args.dataset, args.output, args.cache, args.camera)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
