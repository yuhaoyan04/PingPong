"""Compare old and padded ball ROIs and conservative spatial filtering."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.event_denoise import neighbor_filter
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def binary_crop(events, path, now, scale):
    radii = path[:, 3] * scale
    surf = trajectory_compensated_ball_patch(events, path[:, 0].astype(np.int64),
                                            path[:, 1:3], radii, 256, now, 500, 1000)
    return (np.max(surf, axis=0) * 255 > 45).astype(np.uint8) * 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--pipeline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--crop-scale', type=float, default=1.35)
    p.add_argument('--neighbor-radius', type=int, default=3)
    p.add_argument('--min-neighbors', type=int, default=2)
    args = p.parse_args()
    if args.crop_scale < 1: raise ValueError('crop scale must be >= 1')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summary = json.loads((args.pipeline/'summary.json').read_text(encoding='utf8'))
    readers = {}
    for item in summary['groups']:
        name = item['group']; track_file = args.pipeline/name/'raw_refined_path.npz'
        if not track_file.exists(): continue
        meta = json.loads((args.groups/name/'source.json').read_text(encoding='utf8'))
        if meta['raw'] not in readers: readers[meta['raw']] = RawWindowReader(meta['raw'])
        reader = readers[meta['raw']]
        records = np.load(track_file)['records']; path = records[:, :4]
        if len(path) < 2: continue
        events = reader.window(int(path[0, 0])-1000, int(path[-1, 0])+1)
        events = events[np.argsort(events['timestamp'], kind='stable')]
        times = events['timestamp'].astype(np.int64, copy=False)
        group = args.output/name; group.mkdir()
        writer = cv2.VideoWriter(str(group/'crop_comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1024, 310))
        if not writer.isOpened(): raise RuntimeError('could not open comparison video')
        try:
            for i, row in enumerate(records):
                now = int(row[0]); lo = np.searchsorted(times, now-1000); hi = np.searchsorted(times, now)
                current = events[lo:hi]
                original = binary_crop(current, path, now, 1.0)
                expanded = binary_crop(current, path, now, args.crop_scale)
                filtered = neighbor_filter(expanded, args.neighbor_radius, args.min_neighbors)
                removed = cv2.cvtColor(expanded, cv2.COLOR_GRAY2BGR)
                removed[(expanded != 0) & (filtered == 0)] = (0, 0, 255)
                canvas = np.zeros((310, 1024, 3), np.uint8)
                for k, panel in enumerate((cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
                                           cv2.cvtColor(expanded, cv2.COLOR_GRAY2BGR),
                                           cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), removed)):
                    canvas[28:284, k*256:(k+1)*256] = panel
                for k, title in enumerate(('OLD CROP', 'PADDED CROP', 'FILTERED', 'RED = REMOVED')):
                    cv2.putText(canvas, title, (k*256+5, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
                status = 'SUPPORTED' if row[5] else 'FALLBACK: NOT VERIFIED BALL'
                cv2.putText(canvas, f'{name} {now} us  {status}  ROI x{args.crop_scale:.2f}', (5, 303), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
                writer.write(canvas)
                if i == len(records)//2: cv2.imwrite(str(group/'comparison_middle.png'), canvas)
        finally: writer.release()
        print(name, len(records), flush=True)
    (args.output/'README.txt').write_text(
        f'Unverified review only. Crop radius x{args.crop_scale}; this is a margin, not a measured radius. '
        f'Filter radius={args.neighbor_radius}, min other occupied pixels={args.min_neighbors}. '
        'Red shows deleted pixels. Filtering may delete real marker events. Fallback is not a detection.\n', encoding='utf8')


if __name__ == '__main__': main()
