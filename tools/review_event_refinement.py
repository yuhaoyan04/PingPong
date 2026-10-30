"""Review experimental event-warp circle refinement, without touching labels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.ball_event_refine import refine_track, refine_track_adaptive
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def to_binary(events, path, now):
    surface = trajectory_compensated_ball_patch(events, path[:, 0].astype(np.int64),
                                               path[:, 1:3], path[:, 3], 256, int(now), 500, 1000)
    return (np.max(surface, axis=0)*255 > 45).astype(np.uint8)*255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--group', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--adaptive-radius', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f'refusing to overwrite {args.output}')
    cv2.setNumThreads(1)
    source = json.loads((args.group/'source.json').read_text(encoding='utf-8'))
    coarse = np.load(args.review/'candidate_tracks.npz')['new']
    if len(coarse) < 4:
        raise ValueError('coarse candidate has fewer than four measurements')
    reader = RawWindowReader(source['raw'])
    events = reader.window(int(coarse[0, 0])-1000, int(coarse[-1, 0])+1)
    records, inliers = (refine_track_adaptive if args.adaptive_radius else refine_track)(events, coarse)
    if len(records) < 2: raise ValueError('refinement produced insufficient time samples')
    ordered = events[np.argsort(events['timestamp'], kind='stable')]
    timestamps = ordered['timestamp'].astype(np.int64, copy=False)
    seed_path = np.column_stack((records[:, 0], records[:, 6:8], records[:, 3]))
    new_path = records[:, :4]
    args.output.mkdir(parents=True)
    writer = cv2.VideoWriter(str(args.output/'comparison_raw_1ms.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 25, (1160, 400))
    if not writer.isOpened(): raise RuntimeError('video writer unavailable')
    try:
        for i, row in enumerate(records):
            now = int(row[0]); lo = np.searchsorted(timestamps, now-1000); hi = np.searchsorted(timestamps, now)
            sample = ordered[lo:hi]
            frame = np.full((reader.info.height, reader.info.width), 127, np.uint8)
            valid = (sample['x'] < reader.info.width) & (sample['y'] < reader.info.height)
            sample_valid = sample[valid]
            if len(sample_valid): frame[sample_valid['y'], sample_valid['x']] = np.where(sample_valid['polarity'], 255, 0)
            full = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_GRAY2BGR)
            seed_center = tuple(np.rint(row[6:8]*.5).astype(int))
            cv2.circle(full, seed_center, max(1, round(row[3]*.5)), (0, 255, 0), 1)
            supported = bool(row[5])
            if supported:
                cv2.circle(full, tuple(np.rint(row[1:3]*.5).astype(int)), max(1, round(row[3]*.5)), (255, 0, 255), 1)
            seed_img = to_binary(sample, seed_path, now)
            refined_img = np.zeros((256, 256), np.uint8)
            prior = i > 0 and records[i-1, 0] == now-1000 and bool(records[i-1, 5])
            if supported and prior:
                refined_img = to_binary(sample, new_path, now)
            else:
                cv2.putText(refined_img, 'INSUFFICIENT SUPPORT', (8, 130), cv2.FONT_HERSHEY_SIMPLEX, .42, 180, 1)
            canvas = np.zeros((400, 1160, 3), np.uint8)
            canvas[:360, :640] = full
            canvas[30:286, 648:904] = cv2.cvtColor(seed_img, cv2.COLOR_GRAY2BGR)
            canvas[30:286, 904:1160] = cv2.cvtColor(refined_img, cv2.COLOR_GRAY2BGR)
            for x, text in [(8, 'SAME RAW WINDOW: GREEN SEED / MAGENTA REFINED'),
                            (650, 'SMOOTH SEED CROP'), (907, 'RAW REFINED CROP')]:
                cv2.putText(canvas, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            cv2.putText(canvas, f'{args.group.name} RAW {now} us | {"SUPPORTED" if supported else "FALLBACK: NO OBSERVED CIRCLE"} score={row[4]:.2f}',
                        (8, 384), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
            writer.write(canvas)
            if i in {len(records)//4, len(records)//2, 3*len(records)//4}:
                cv2.imwrite(str(args.output/f'frame_{i:03d}.png'), canvas)
    finally:
        writer.release()
    np.savez_compressed(args.output/'raw_refined_path.npz', records=records, coarse=coarse, coarse_inliers=inliers)
    info = {'group': args.group.name, 'samples': len(records), 'step_us': 1000, 'window_us': 1000,
            'adaptive_radius': args.adaptive_radius,
            'supported': int(np.count_nonzero(records[:, 5])), 'fallback': int(np.count_nonzero(records[:, 5] == 0)),
            'coarse_observations': len(coarse), 'coarse_inliers': int(inliers.sum()),
            'status': 'experimental_unverified',
            'warning': 'Event ring support is a heuristic, not ground truth. Fallback is not an observed ball.'}
    (args.output/'report.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__': main()
