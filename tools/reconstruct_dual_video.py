"""Reconstruct two time-aligned RAW views as slow event-accumulation video."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from pingpong_event_pipeline.events import RawWindowReader


def frame(events, start, end, width, height, native):
    picture = np.full((height, width), 127, np.uint8)
    if events.size:
        t = events['timestamp'].astype(np.int64, copy=False)
        valid = ((t >= start) & (t < end) & (events['x'] < width) & (events['y'] < height))
        events = events[valid]
        x, y = events['x'].astype(np.intp), events['y'].astype(np.intp)
        picture[y, x] = np.where(events['polarity'], 255, 0).astype(np.uint8)
    return picture if native else cv2.resize(picture, (640, 360), interpolation=cv2.INTER_NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--master-raw', required=True)
    p.add_argument('--slave-raw', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--window-us', type=int, required=True)
    p.add_argument('--step-us', type=int, required=True)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--contact-sheet', action='store_true')
    p.add_argument('--native-size', action='store_true')
    p.add_argument('--start-s', type=float, default=0.0, help='start relative to common recording origin')
    p.add_argument('--end-s', type=float, help='end relative to common recording origin')
    args = p.parse_args()
    if min(args.window_us, args.step_us, args.fps) <= 0:
        p.error('window, step and fps must be positive')
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    origin = max(r.info.start_us for r in readers)
    end = min(r.info.end_us for r in readers)
    if args.start_s < 0 or (args.end_s is not None and args.end_s <= args.start_s):
        p.error('invalid clip interval')
    clip_start = origin + round(args.start_s * 1e6)
    clip_end = min(end, origin + round(args.end_s * 1e6)) if args.end_s is not None else end
    if clip_start + args.window_us >= clip_end:
        p.error('clip interval does not overlap the common recording')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.native_size and (readers[0].info.width != readers[1].info.width or readers[0].info.height != readers[1].info.height):
        raise RuntimeError('Native-size playback requires equal sensor dimensions')
    video_size = ((readers[0].info.width * 2, readers[0].info.height)
                  if args.native_size else (1280, 360))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, video_size)
    if not writer.isOpened():
        raise RuntimeError('Could not open video writer')
    count = 0
    sheet_tiles = []
    caches = [None, None]
    try:
        for timestamp in range(clip_start + args.window_us, clip_end, args.step_us):
            panels = []
            for index, reader in enumerate(readers):
                cache_start = timestamp - args.window_us
                if caches[index] is None or cache_start < caches[index][0] or timestamp > caches[index][1]:
                    cache_end = min(end, timestamp + 100000)
                    data = reader.window(cache_start, cache_end)
                    times = data['timestamp'].astype(np.int64, copy=False)
                    valid = (times >= cache_start) & (times < cache_end)
                    data, times = data[valid], times[valid]
                    if times.size > 1 and np.any(times[1:] < times[:-1]):
                        order = np.argsort(times, kind='stable')
                        data, times = data[order], times[order]
                    caches[index] = (cache_start, cache_end, data, times)
                _, _, data, times = caches[index]
                first = np.searchsorted(times, timestamp - args.window_us, side='left')
                last = np.searchsorted(times, timestamp, side='left')
                events = data[first:last]
                gray = frame(events, timestamp - args.window_us, timestamp, reader.info.width, reader.info.height, args.native_size)
                panels.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            view = np.hstack(panels)
            cv2.putText(view, f'MASTER | SLAVE    t={(timestamp-origin)/1e6:.3f}s   window={args.window_us/1000:g}ms',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1, cv2.LINE_AA)
            writer.write(view)
            if args.contact_sheet:
                sheet_tiles.append(cv2.resize(view, (640, 180), interpolation=cv2.INTER_AREA))
                if len(sheet_tiles) == 30:
                    sheet = np.vstack([np.hstack(sheet_tiles[i:i+6]) for i in range(0, 30, 6)])
                    cv2.imwrite(str(args.output.with_name(f'{args.output.stem}_sheet_{count//30+1}.jpg')), sheet)
                    sheet_tiles.clear()
            count += 1
    finally:
        writer.release()
    print(f'{args.output.resolve()} frames={count} playback_speed={args.fps*args.step_us/1e6:.3f}x')


if __name__ == '__main__':
    main()
