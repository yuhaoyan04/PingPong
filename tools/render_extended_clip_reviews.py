"""Render padded RAW review clips without replacing original clips or labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from pingpong_event_pipeline.events import RawWindowReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pre-ms', type=int, default=200)
    p.add_argument('--post-ms', type=int, default=300)
    args = p.parse_args()
    if min(args.pre_ms, args.post_ms) < 0: raise ValueError('padding must be nonnegative')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    readers = {}; results = []
    for number in range(1, 10):
        metas = [json.loads((args.groups/f'serve{number:02d}_{camera}'/'source.json').read_text(encoding='utf8'))
                 for camera in ('F3E', 'F59')]
        if any(m['video_first_us'] != metas[0]['video_first_us'] or m['video_step_us'] != metas[0]['video_step_us'] for m in metas):
            raise ValueError('camera video metadata differs')
        cap = cv2.VideoCapture(metas[0]['video'])
        if not cap.isOpened(): raise RuntimeError('original clip not readable')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if n < 1: raise ValueError('empty source clip')
        first = int(metas[0]['video_first_us']); step = int(metas[0]['video_step_us'])
        old_end = first + n*step
        pair = []
        for m in metas:
            if m['raw'] not in readers: readers[m['raw']] = RawWindowReader(m['raw'])
            pair.append(readers[m['raw']])
        start = max(first-args.pre_ms*1000, *(r.info.start_us+1000 for r in pair))
        end = min(old_end+args.post_ms*1000, *(r.info.end_us for r in pair))
        if end <= start: raise ValueError('no RAW data in padded interval')
        samples = []
        for reader in pair:
            e = reader.window(start-1000, end+1)
            e = e[np.argsort(e['timestamp'], kind='stable')]
            samples.append((e, e['timestamp'].astype(np.int64, copy=False)))
        video = args.output/f'serve_{number:02d}_extended.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 400))
        if not writer.isOpened(): raise RuntimeError('could not open output video')
        try:
            for t in range(start, end, step):
                canvas = np.zeros((400, 1280, 3), np.uint8)
                for k, ((e, ts), reader) in enumerate(zip(samples, pair)):
                    frame = np.full((reader.info.height, reader.info.width, 3), 127, np.uint8)
                    w = e[np.searchsorted(ts,t-1000):np.searchsorted(ts,t)]
                    valid = (w['x'] < reader.info.width) & (w['y'] < reader.info.height)
                    w = w[valid]; polarity = w['polarity'].astype(bool)
                    frame[w['y'][~polarity], w['x'][~polarity]] = (255, 0, 0)
                    frame[w['y'][polarity], w['x'][polarity]] = (0, 255, 255)
                    canvas[:360, k*640:(k+1)*640] = cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
                    cv2.putText(canvas, ('F3E' if k == 0 else 'F59'), (k*640+8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
                part = 'PRE-PADDING' if t < first else ('ORIGINAL RANGE' if t < old_end else 'POST-PADDING')
                cv2.putText(canvas, f'serve {number:02d} | {part} | RAW timestamp {t} us | window 1 ms', (8,385), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                writer.write(canvas)
        finally: writer.release()
        results.append({'serve': number, 'original_start_us': first, 'original_end_exclusive_us': old_end,
                        'extended_start_us': start, 'extended_end_exclusive_us': end,
                        'video': video.name, 'status': 'manual_review_required'})
        print(video.name, flush=True)
    (args.output/'manifest.json').write_text(json.dumps({'clips': results,
        'warning': 'Padding does not establish ball visibility or verified stereo synchronization. '
                   'Frames are sampled at original clip step, each from a 1ms RAW window.'}, indent=2), encoding='utf8')


if __name__ == '__main__': main()
