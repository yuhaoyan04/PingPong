"""One-view event trajectory and anonymous marker-proposal review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae, trajectory_compensated_ball_patch
from propose_physical_marker_centers import polarity_midpoints
from review_real_serve_markers import colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window-us', type=int, default=1000)
    parser.add_argument('--step-us', type=int, default=2000)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    data = np.load(args.path)
    ts = data['timestamps_us'].astype(np.int64)
    center = data['centers_px'].astype(np.float64)
    radius = data['radii_px'].astype(np.float64)
    if args.window_us <= 0 or args.step_us <= 0 or np.any(data['measured_scores'] <= 0):
        raise ValueError('invalid interval or unobserved trajectory bins')
    reader = RawWindowReader(args.raw)
    events = reader.window(int(ts[0]-args.window_us), int(ts[-1])+1)
    events = events[np.argsort(events['timestamp'], kind='stable')]
    event_ts = events['timestamp'].astype(np.int64, copy=False)
    samples = np.arange(ts[0] + args.window_us, ts[-1], args.step_us, dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_dir/'trajectory_and_markers.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 25, (1360, 400))
    if not writer.isOpened():
        raise RuntimeError('video writer failed')
    logs = []
    all_steps = np.linspace(ts[0], ts[-1], 100)
    path_xy = np.column_stack([np.interp(all_steps, ts, center[:, ax]) for ax in (0, 1)])
    path_draw = np.rint(path_xy * .5).astype(np.int32).reshape(-1, 1, 2)
    try:
        for i, t in enumerate(samples):
            lo = np.searchsorted(event_ts, t-args.window_us, side='left')
            hi = np.searchsorted(event_ts, t, side='left')
            current = events[lo:hi]
            ref_c = np.array([np.interp(t, ts, center[:, ax]) for ax in (0, 1)])
            ref_r = float(np.interp(t, ts, radius))
            full = cv2.resize(render_bgr(sae(current, reader.info.width, reader.info.height,
                                            int(t), args.window_us/2)), (640, 360), interpolation=cv2.INTER_AREA)
            cv2.polylines(full, [path_draw], False, (0, 150, 0), 1, cv2.LINE_AA)
            ref_draw = tuple(np.rint(ref_c * .5).astype(int))
            cv2.circle(full, ref_draw, max(1, round(ref_r*.5)), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(full, ref_draw, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            surf = trajectory_compensated_ball_patch(current, ts, center, radius, 256,
                                                     int(t), args.window_us/2, args.window_us)
            original = colored(surf)
            candidates = polarity_midpoints(original, 35, 40, .36, 4)
            candidates = [p for p in candidates if p['assigned_active_pixels'] >= 2
                          and 1.5 <= p['polarity_separation_px'] <= 40]
            annotated = original.copy()
            for j, p in enumerate(candidates, 1):
                pt = (round(p['x']), round(p['y']))
                cv2.drawMarker(annotated, pt, (0, 255, 0), cv2.MARKER_CROSS, 13, 1)
                cv2.circle(annotated, pt, 8, (0, 255, 0), 1)
                cv2.putText(annotated, str(j), (pt[0]+6, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (0, 255, 0), 1, cv2.LINE_AA)
            for tile in (original, annotated):
                cv2.circle(tile, (128, 128), 127, (80, 80, 80), 1)
            cropped = cv2.resize(original, (360, 360), interpolation=cv2.INTER_NEAREST)
            marked = cv2.resize(annotated, (360, 360), interpolation=cv2.INTER_NEAREST)
            frame = np.zeros((400, 1360, 3), np.uint8)
            frame[0:360, 0:640] = full
            frame[0:360, 640:1000] = cropped
            frame[0:360, 1000:1360] = marked
            for pos, label in [((8, 20), 'FULL VIEW + IMAGE TRACK'), ((646, 20), 'BALL NORMALIZED'),
                               ((1006, 20), 'GREEN = CANDIDATES, NOT GT')]:
                cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f't={(t-ts[0])/1000:.1f}ms | window={args.window_us/1000:g}ms | blue=OFF yellow=ON | IDs not tracked',
                        (8, 384), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if i in {0, len(samples)//2, len(samples)-1}:
                cv2.imwrite(str(args.output_dir / f'review_{i:03d}.png'), frame)
            ev_t = current['timestamp'].astype(np.int64, copy=False)
            c_x = np.interp(ev_t, ts, center[:, 0]); c_y = np.interp(ev_t, ts, center[:, 1])
            r = np.interp(ev_t, ts, radius)
            dist = np.hypot(current['x'].astype(float)-c_x, current['y'].astype(float)-c_y)/r
            p = current['polarity'] != 0
            counts = {}
            for name, mask in [('inner', dist < .78), ('rim', (dist >= .78) & (dist <= 1.05)),
                               ('outside', dist > 1.3)]:
                counts[name] = {'on': int(np.count_nonzero(mask & p)), 'off': int(np.count_nonzero(mask & ~p))}
            logs.append({'index': i, 'timestamp_us': int(t), 'center_px': ref_c.tolist(),
                         'radius_px': ref_r, 'candidate_points_patch_px': candidates, 'event_counts': counts})
    finally:
        writer.release()
    payload = {'source_raw': str(Path(args.raw).resolve()), 'source_path': str(args.path.resolve()),
               'fps': 25, 'step_us': args.step_us, 'window_us': args.window_us,
               'note': 'Candidate labels restart each frame; not marker IDs or truth. Ball track is provisional.',
               'frames': logs}
    (args.output_dir/'review.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('output', args.output_dir.resolve(), 'frames', len(logs), flush=True)


if __name__ == '__main__':
    main()
