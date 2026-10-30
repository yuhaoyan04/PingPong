"""Export ball-centred raw-event binary crops for manual annotation."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--path', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--group', required=True)
    p.add_argument('--window-us', type=int, default=1000)
    p.add_argument('--step-us', type=int, default=500)
    p.add_argument('--threshold', type=int, default=45)
    args = p.parse_args()
    if not 0 <= args.threshold <= 255 or min(args.window_us, args.step_us) <= 0:
        p.error('invalid threshold or time interval')
    out = args.output / args.group
    binary_dir, polarity_dir = out / 'binary', out / 'polarity_reference'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing dataset or annotations: {out}')
    binary_dir.mkdir(parents=True)
    polarity_dir.mkdir(parents=True)
    track = np.load(args.path)
    t = track['timestamps_us'].astype(np.int64)
    centers, radii = track['centers_px'].astype(float), track['radii_px'].astype(float)
    if np.any(track['measured_scores'] <= 0):
        raise ValueError('path contains unobserved detection bins; review path before export')
    reader = RawWindowReader(str(args.raw))
    events = reader.window(int(t[0]) - args.window_us, int(t[-1]) + 1)
    events = events[np.argsort(events['timestamp'], kind='stable')]
    event_times = events['timestamp'].astype(np.int64, copy=False)
    rows = []
    for index, now in enumerate(np.arange(t[0] + args.window_us, t[-1], args.step_us, dtype=np.int64)):
        lo = np.searchsorted(event_times, now - args.window_us, side='left')
        hi = np.searchsorted(event_times, now, side='left')
        surf = trajectory_compensated_ball_patch(events[lo:hi], t, centers, radii, 256, int(now),
                                                  args.window_us / 2, args.window_us)
        off, on = surf[0] * 255 > args.threshold, surf[1] * 255 > args.threshold
        image = ((off | on) * 255).astype(np.uint8)
        reference = np.zeros((256, 256, 3), np.uint8)
        reference[off, 0] = 255
        reference[on, 1] = 255
        reference[on, 2] = 255
        name = f'{args.group}_{index:05d}.png'
        if not cv2.imwrite(str(binary_dir / name), image) or not cv2.imwrite(str(polarity_dir / name), reference):
            raise RuntimeError(f'failed to write {name}')
        rows.append({'group': args.group, 'image': f'binary/{name}', 'polarity_reference': f'polarity_reference/{name}',
                     'frame_index': index, 'timestamp_us': int(now), 'window_us': args.window_us,
                     'center_x_px': float(np.interp(now, t, centers[:, 0])),
                     'center_y_px': float(np.interp(now, t, centers[:, 1])),
                     'radius_px': float(np.interp(now, t, radii))})
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'source.json').write_text(json.dumps({'raw': str(args.raw.resolve()), 'path': str(args.path.resolve()),
        'group': args.group, 'frames': len(rows), 'window_us': args.window_us, 'step_us': args.step_us,
        'binary_threshold': args.threshold, 'time_surface_tau_us': args.window_us / 2,
        'note': 'binary OR of ON/OFF event time surfaces; unannotated images are not negative labels'}, indent=2), encoding='utf-8')
    print(f'{out.resolve()}: {len(rows)} frames')


if __name__ == '__main__':
    main()
