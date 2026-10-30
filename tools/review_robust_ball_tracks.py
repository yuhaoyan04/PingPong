"""Compare legacy and motion-consistent ball hypotheses without changing annotations."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from batch_marker_crops import candidates
from pingpong_event_pipeline.ball_track_robust import select_track
from pingpong_event_pipeline.detector import HoughCircleBallDetector
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def supported_circle(timestamp, points, max_gap_us=10000):
    if len(points) < 2 or timestamp < points[0, 0] or timestamp > points[-1, 0]:
        return None
    right = int(np.searchsorted(points[:, 0], timestamp, side='left'))
    if right < len(points) and points[right, 0] == timestamp:
        return points[right, 1:4]
    if right == 0 or right == len(points) or points[right, 0]-points[right-1, 0] > max_gap_us:
        return None
    return np.array([np.interp(timestamp, points[:, 0], points[:, j]) for j in (1, 2, 3)])


def patch(events, points, timestamp):
    result = trajectory_compensated_ball_patch(events, points[:, 0].astype(np.int64),
                points[:, 1:3], points[:, 3], 256, int(timestamp), 500, 1000)
    return (np.max(result, axis=0)*255 > 45).astype(np.uint8)*255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    cv2.setNumThreads(1)
    source = json.loads((args.group/'source.json').read_text(encoding='utf-8'))
    video = cv2.VideoCapture(source['video'])
    if not video.isOpened(): raise ValueError('source video unavailable')
    frames, candidates_by_frame = [], []
    while True:
        ok, frame = video.read()
        if not ok: break
        panel = frame[:, :1280] if source['camera'] == 'F3E' else frame[:, 1280:]
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        activity = np.abs(gray.astype(np.int16)-127).clip(0, 255).astype(np.uint8)
        found = []
        for circle in candidates(panel, source['camera']):
            score = HoughCircleBallDetector._ring_score(activity, *circle)
            if score >= .18:
                found.append((*circle, score))
        candidates_by_frame.append(found)
        frames.append(panel.copy())
    video.release()
    hypothesis = select_track(candidates_by_frame)
    old = np.array(source['source_circle_hypotheses'], dtype=float).reshape(-1, 4)
    first = int(source['video_first_us']); step = int(source['video_step_us'])
    if hypothesis:
        new = np.array([[first+i*step, *circle[:3]] for i, circle in hypothesis['members']], dtype=float)
    else:
        new = np.empty((0, 4), dtype=float)
    reader = RawWindowReader(source['raw'])
    start, end = first-1000, first+len(frames)*step
    events = reader.window(start, end)
    events = events[np.argsort(events['timestamp'], kind='stable')]
    timestamps = events['timestamp'].astype(np.int64, copy=False)
    args.output.mkdir(parents=True)
    writer = cv2.VideoWriter(str(args.output/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1160, 400))
    if not writer.isOpened(): raise RuntimeError('video writer unavailable')
    try:
        for index, frame in enumerate(frames):
            now = first + index*step
            old_c, new_c = supported_circle(now, old), supported_circle(now, new)
            old_view = np.zeros((256, 256), np.uint8); new_view = old_view.copy()
            lo, hi = np.searchsorted(timestamps, now-1000, side='left'), np.searchsorted(timestamps, now, side='left')
            sample = events[lo:hi]
            if old_c is not None:
                cv2.circle(frame, tuple(np.rint(old_c[:2]).astype(int)), round(old_c[2]), (0, 255, 0), 2)
                old_view = patch(sample, old, now)
            if new_c is not None:
                cv2.circle(frame, tuple(np.rint(new_c[:2]).astype(int)), round(new_c[2]), (255, 0, 255), 2)
                new_view = patch(sample, new, now)
            canvas = np.zeros((400, 1160, 3), np.uint8)
            canvas[:360, :640] = cv2.resize(frame, (640, 360))
            canvas[30:286, 648:904] = cv2.cvtColor(old_view, cv2.COLOR_GRAY2BGR)
            canvas[30:286, 904:1160] = cv2.cvtColor(new_view, cv2.COLOR_GRAY2BGR)
            for x, text in [(8, 'GREEN=OLD MAGENTA=NEW'), (650, 'OLD RAW CROP'), (908, 'NEW RAW CROP')]:
                cv2.putText(canvas, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            cv2.putText(canvas, f'{args.group.name} frame={index} t={now}us | BOTH UNVERIFIED', (8, 384),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            writer.write(canvas)
            if index in {len(frames)//3, len(frames)//2, 2*len(frames)//3}:
                cv2.imwrite(str(args.output/f'frame_{index:03d}.png'), canvas)
    finally:
        writer.release()
    np.savez_compressed(args.output/'candidate_tracks.npz', old=old, new=new)
    info = {'group': args.group.name, 'frames': len(frames), 'candidate_count': len(new),
            'old_detection_count': len(old), 'status': 'unverified_candidate' if len(new) else 'no_new_candidate',
            'source_video_step_us': step, 'window_us': 1000,
            'warning': 'No ground truth; both tracks and timing alignment require review. No annotations modified.'}
    if hypothesis:
        info.update({k: v for k, v in hypothesis.items() if k != 'members'})
    (args.output/'report.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__': main()
