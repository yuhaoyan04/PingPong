"""Check saved LED-board snapshots without producing calibration parameters."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from live_stereo_view import make_blob_detector
from manual_calibration_capture import detect_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    detector = make_blob_detector()
    report = []
    for metadata_path in sorted(args.directory.glob('pair_*.json')):
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        item = {'pair': meta['index'], 'views': {}}
        for role in ('master', 'slave'):
            img = cv2.imread(str(args.directory / meta['images'][role]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                item['views'][role] = {'found': False, 'error': 'image missing'}
                continue
            points = detect_grid(img, detector)
            view = {'found': points is not None}
            if points is not None:
                view['center_px'] = np.mean(points, axis=0).round(1).tolist()
                view['bounds_px'] = np.ptp(points, axis=0).round(1).tolist()
                view['points_px'] = points.round(2).tolist()
            item['views'][role] = view
        item['preview_host_end_delta_ms'] = round(abs(
            meta['preview_windows']['master']['host_monotonic'] -
            meta['preview_windows']['slave']['host_monotonic']) * 1000, 1)
        report.append(item)
        print(f"{item['pair']:03d} " + ' '.join(
            f"{role}={'OK ' + str(item['views'][role]['center_px']) + ' ' + str(item['views'][role]['bounds_px']) if item['views'][role]['found'] else 'NOT FOUND'}"
            for role in ('master', 'slave')) + f" host_delta_ms={item['preview_host_end_delta_ms']}", flush=True)
    output = args.directory / 'detection_audit.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
