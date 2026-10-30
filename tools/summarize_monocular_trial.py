"""Create a gated measurement report from an experimental image-space track."""
import argparse
import json
from pathlib import Path

import numpy as np
from pingpong_event_pipeline.measurement_gate import image_motion_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.path)
    result = image_motion_summary(data['timestamps_us'], data['centers_px'], data['radii_px'],
                                  data['measured_scores'])
    result['track_file'] = str(args.path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
