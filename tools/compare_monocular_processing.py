"""Compare event windows and provisional ball translation compensation on one RAW."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_ball_event_filters import stc_cut_trail_mask
from pingpong_event_pipeline.events import RawWindowReader


def panel(events, end, window, center, reference_center, width=180, compensate=False, path=None):
    t = events['timestamp'].astype(np.int64, copy=False)
    chosen = (t >= end-window) & (t < end)
    ev = events[chosen]
    times = t[chosen]
    x, y = ev['x'].astype(np.float64), ev['y'].astype(np.float64)
    if compensate and len(times):
        x += reference_center[0] - np.interp(times, path[0], path[1][:, 0])
        y += reference_center[1] - np.interp(times, path[0], path[1][:, 1])
    x = np.rint(x - center[0] + width/2).astype(np.int32)
    y = np.rint(y - center[1] + width/2).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < width)
    image = np.full((width, width, 3), 20, np.uint8)
    pol = ev['polarity'][valid] != 0
    x, y = x[valid], y[valid]
    image[y[~pol], x[~pol]] = (255, 90, 20)
    image[y[pol], x[pol]] = (20, 230, 255)
    return image, int(np.count_nonzero(valid))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threshold-us', type=int, default=2000)
    args = p.parse_args()
    track = np.load(args.path)
    times = track['timestamps_us'].astype(np.int64)
    centers = track['centers_px'].astype(float)
    reader = RawWindowReader(args.raw)
    events = reader.window(int(times[0]-15000), int(times[-1])+1)
    order = np.argsort(events['timestamp'], kind='stable')
    events = events[order]
    x = events['x'].astype(np.int32); y = events['y'].astype(np.int32)
    t = events['timestamp'].astype(np.int64); polarity = events['polarity'].astype(np.int8)
    valid = (x >= 0) & (x < reader.info.width) & (y >= 0) & (y < reader.info.height)
    if not np.all(valid):
        raise RuntimeError('Out of bounds events')
    keep = stc_cut_trail_mask(x, y, t, polarity, reader.info.width, reader.info.height, args.threshold_us)
    filtered = events[keep]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    output = []
    for sample_no, end in enumerate(np.linspace(times[0]+5000, times[-1]-1000, 3, dtype=np.int64), 1):
        ref = np.array([np.interp(end, times, centers[:, axis]) for axis in range(2)])
        rows.clear()
        for window in (500, 1000, 2000, 5000):
            cols = []
            for label, stream, comp in [('RAW', events, False), ('WARP', events, True), ('WARP+STC', filtered, True)]:
                tile, count = panel(stream, end, window, ref, ref, compensate=comp, path=(times, centers))
                tile = cv2.resize(tile, (360, 360), interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, f'{label} {window/1000:g}ms n={count}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            .47, (255, 255, 255), 1, cv2.LINE_AA)
                cols.append(tile)
                output.append({'sample': sample_no, 'end_us': int(end), 'window_us': window,
                               'method': label, 'events_in_crop': count})
            rows.append(np.hstack(cols))
        cv2.imwrite(str(args.output_dir / f'compare_{sample_no:02d}.png'), np.vstack(rows))
    video_path = args.output_dir / 'comparison_slow.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1080, 1440))
    if not writer.isOpened():
        raise RuntimeError('Could not create comparison video')
    try:
        for end in np.arange(times[0]+5000, times[-1], 2000, dtype=np.int64):
            ref = np.array([np.interp(end, times, centers[:, axis]) for axis in range(2)])
            video_rows = []
            for window in (500, 1000, 2000, 5000):
                cols = []
                for label, stream, comp in [('RAW', events, False), ('WARP', events, True), ('WARP+STC', filtered, True)]:
                    tile, count = panel(stream, end, window, ref, ref, compensate=comp, path=(times, centers))
                    tile = cv2.resize(tile, (360, 360), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile, f'{label} {window/1000:g}ms n={count}', (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
                    cols.append(tile)
                video_rows.append(np.hstack(cols))
            writer.write(np.vstack(video_rows))
    finally:
        writer.release()
    report = {'raw': str(Path(args.raw).resolve()), 'path': str(args.path.resolve()),
              'filter': f'experimental pixel STC cut trail, {args.threshold_us} us; not hardware equivalent',
              'warmup_us': 15000, 'warning': 'translation-only warp based on provisional 2D track; no spin estimate; counts cover square crop',
              'video': str(video_path.resolve()), 'video_step_us': 2000, 'video_fps': 25,
              'samples': output}
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
