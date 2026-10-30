"""Audit detected 3x4 LED grids; diagnostic only, not calibration acceptance."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads((args.session / 'detection_audit.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    grid = np.array([(x, y) for y in range(4) for x in range(3)], np.float32)
    result = []
    for pair in data:
        item = {'pair': pair['pair'], 'views': {}}
        panels = []
        for role in ('master', 'slave'):
            view = pair['views'][role]
            image = cv2.imread(str(args.session / f"{role}_{pair['pair']:03d}.png"))
            if image is None:
                continue
            summary = {'found': view['found']}
            if view['found']:
                pts = np.asarray(view['points_px'], np.float32)
                matrix, _ = cv2.findHomography(grid, pts, method=0)
                if matrix is not None:
                    predicted = cv2.perspectiveTransform(grid.reshape(1, -1, 2), matrix).reshape(-1, 2)
                    residuals = np.linalg.norm(predicted - pts, axis=1)
                    summary.update(homography_rms_px=float(np.sqrt(np.mean(residuals ** 2))),
                                   homography_max_px=float(np.max(residuals)),
                                   border_min_px=float(np.min(np.column_stack((pts[:, 0], pts[:, 1],
                                       image.shape[1]-1-pts[:, 0], image.shape[0]-1-pts[:, 1])))))
                for index, (x, y) in enumerate(pts):
                    point = (int(round(x)), int(round(y)))
                    cv2.circle(image, point, 9, (0, 0, 255), 2)
                    cv2.putText(image, str(index + 1), (point[0] + 8, point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            item['views'][role] = summary
            panels.append(image)
        if len(panels) == 2:
            joined = np.hstack(panels)
            cv2.imwrite(str(args.output / f"pair_{pair['pair']:03d}_numbered.jpg"), joined)
        result.append(item)
    (args.output / 'point_geometry.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for item in result:
        values = [item['views'].get(role, {}).get('homography_rms_px') for role in ('master', 'slave')]
        print(item['pair'], ['%.2f' % v if v is not None else 'NOT FOUND' for v in values])


if __name__ == '__main__':
    main()
