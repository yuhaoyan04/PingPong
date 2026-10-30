"""Estimate review-only ball tracks in 5 ms clips and export RAW event crops.

Timecodes are manually read from the burned-in first frame of each clip.  These
tracks are unverified hypotheses; the exported dataset is NOT ground truth.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


FIRST_SECONDS = [3.351, 6.751, 9.051, 12.501, 14.251, 17.351, 19.751, 22.651, 25.151]


def candidates(panel, camera):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    activity = np.abs(gray.astype(np.int16) - 127).clip(0, 255).astype(np.uint8)
    small = cv2.GaussianBlur(cv2.resize(activity, (320, 180)), (5, 5), 0)
    min_r, max_r = ((5, 17) if camera == 'F3E' else (8, 27))
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, 1, 12, param1=35,
                               param2=10, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []
    return [tuple(map(float, c * 4)) for c in circles[0, :15]]


def associate(frames, gate=50.):
    tracks = []
    for index, circles in enumerate(frames):
        alive = [s for s in tracks if index - s['last'] <= 2]
        costs = np.full((len(alive), len(circles) + len(alive)), gate, dtype=float)
        for i, s in enumerate(alive):
            dt = index - s['last']
            pred = np.array(s['circle'][:2]) + np.array(s['velocity']) * dt
            for j, c in enumerate(circles):
                d = float(np.linalg.norm(pred - c[:2]))
                ratio = max(c[2], s['circle'][2]) / min(c[2], s['circle'][2])
                if d < gate and ratio < 1.45:
                    costs[i, j] = d + 8 * abs(np.log(ratio))
        used = set()
        if alive and circles:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if j >= len(circles) or costs[i, j] >= gate: continue
                s, c = alive[i], circles[j]
                dt = index - s['last']
                v = (np.array(c[:2]) - s['circle'][:2]) / dt
                s['velocity'] = (.5*np.array(s['velocity']) + .5*v).tolist()
                s['last'], s['circle'] = index, c
                s['members'].append((index, c)); used.add(int(j))
        for j, circle in enumerate(circles):
            if j not in used:
                tracks.append({'last': index, 'circle': circle, 'velocity': [0., 0.], 'members': [(index, circle)]})
    eligible = []
    for track in tracks:
        pts = track['members']
        if len(pts) >= 4:
            displacement = np.linalg.norm(np.array(pts[-1][1][:2]) - pts[0][1][:2])
            if displacement >= 15:
                eligible.append(track)
    return max(eligible, key=lambda s: len(s['members']), default=None)


def export(group, reader, track, first_us, clip_path, camera):
    points = track['members']
    detection_t = np.asarray([first_us + i*5000 for i, _ in points], np.int64)
    circles = np.asarray([c for _, c in points], float)
    start, end = int(detection_t[0]), int(detection_t[-1])
    data = reader.window(start-1000, end+1)
    data = data[np.argsort(data['timestamp'], kind='stable')]
    evt = data['timestamp'].astype(np.int64, copy=False)
    binary, polarity = group/'binary', group/'polarity_reference'
    if group.exists():
        raise FileExistsError(f'refusing to overwrite {group}')
    binary.mkdir(parents=True); polarity.mkdir()
    rows = []
    for now in np.arange(start+1000, end, 500, dtype=np.int64):
        right = int(np.searchsorted(detection_t, now, side='right'))
        if right == 0 or right == len(detection_t) or detection_t[right] - detection_t[right-1] > 10000:
            continue
        lo, hi = np.searchsorted(evt, now-1000, side='left'), np.searchsorted(evt, now, side='left')
        surf = trajectory_compensated_ball_patch(data[lo:hi], detection_t, circles[:, :2], circles[:, 2], 256, int(now), 500, 1000)
        off, on = surf[0]*255 > 45, surf[1]*255 > 45
        mask = ((off | on)*255).astype(np.uint8)
        color = np.zeros((256, 256, 3), np.uint8); color[off, 0] = 255; color[on, 1:] = 255
        name = f'{group.name}_{len(rows):05d}.png'
        if not cv2.imwrite(str(binary/name), mask) or not cv2.imwrite(str(polarity/name), color):
            raise RuntimeError(f'could not write {name}')
        rows.append({'group': group.name, 'image': f'binary/{name}', 'polarity_reference': f'polarity_reference/{name}',
                     'frame_index': len(rows), 'timestamp_us': int(now), 'window_us': 1000,
                     'center_x_px': float(np.interp(now, detection_t, circles[:, 0])),
                     'center_y_px': float(np.interp(now, detection_t, circles[:, 1])),
                     'radius_px': float(np.interp(now, detection_t, circles[:, 2]))})
    if rows:
        with (group/'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    meta = {'raw': str(Path(reader.path).resolve()) if hasattr(reader, 'path') else None,
            'video': str(clip_path.resolve()), 'camera': camera, 'video_first_us': first_us,
            'video_step_us': 5000, 'frames': len(rows), 'detection_count': len(points),
            'source_circle_hypotheses': [[int(first_us + i*5000), *map(float, c)] for i, c in points],
            'track_status': 'automatic_unreviewed',
            'warning': 'Candidate track from compressed review video. Verify against original before labeling.'}
    (group/'source.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--clips', nargs='*', type=int, default=list(range(1, 10)))
    p.add_argument('--cameras', nargs='+', choices=['F3E', 'F59'], default=['F3E', 'F59'])
    p.add_argument('--gate-px', type=float, default=50.)
    args = p.parse_args()
    cv2.setNumThreads(1)
    readers = {'F3E': RawWindowReader(str(args.capture/'master_DSP01F1FFF3E.raw')),
               'F59': RawWindowReader(str(args.capture/'slave_DSP01F1FFF59.raw'))}
    origin = max(r.info.start_us for r in readers.values())
    args.output.mkdir(parents=True, exist_ok=True)
    report_file = args.output/'batch_report.json'
    results = json.loads(report_file.read_text(encoding='utf-8')).get('results', []) if report_file.exists() else []
    for number in args.clips:
        clip = args.capture/'clips_checked'/f'serve_{number:02d}_1ms.mp4'
        video = cv2.VideoCapture(str(clip))
        if not video.isOpened(): raise ValueError(f'cannot open {clip}')
        frames = {'F3E': [], 'F59': []}
        while True:
            ok, frame = video.read()
            if not ok: break
            if frame.shape[:2] != (720, 2560): raise ValueError(f'unexpected clip size: {frame.shape}')
            frames['F3E'].append(candidates(frame[:, :1280], 'F3E'))
            frames['F59'].append(candidates(frame[:, 1280:], 'F59'))
        video.release()
        first_us = origin + round(FIRST_SECONDS[number-1]*1e6)
        for camera in args.cameras:
            name = f'serve{number:02d}_{camera}'
            results = [row for row in results if row['group'] != name]
            track = associate(frames[camera], args.gate_px)
            if track is None:
                results.append({'group': name, 'status': 'no_automatic_track', 'frames': 0})
                print(name, 'no automatic track', flush=True); continue
            count = export(args.output/name, readers[camera], track, first_us, clip, camera)
            results.append({'group': name, 'status': 'unreviewed', 'frames': count,
                            'detection_count': len(track['members'])})
            print(name, 'frames', count, 'detections', len(track['members']), flush=True)
        report_file.write_text(json.dumps({'origin_us': int(origin), 'results': results}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
