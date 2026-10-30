"""Compare dual RAW event images at matched times and different accumulation windows."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from pingpong_event_pipeline.events import RawWindowReader


def image(events, width, height, start, end):
    result = np.full((height, width), 127, np.uint8)
    if events.size:
        t = events['timestamp'].astype(np.int64, copy=False)
        mask = ((t >= start) & (t < end) & (events['x'] < width) & (events['y'] < height))
        ev = events[mask]
        # Input events are chronological; assignment yields the last event at each pixel.
        result[ev['y'].astype(np.intp), ev['x'].astype(np.intp)] = np.where(ev['polarity'], 255, 0).astype(np.uint8)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--master-raw', required=True)
    p.add_argument('--slave-raw', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--samples', type=int, default=15)
    p.add_argument('--windows-us', type=int, nargs='+', default=[500, 1000, 2000, 5000, 33000])
    args = p.parse_args()
    if args.samples < 1 or any(w <= 0 for w in args.windows_us):
        p.error('samples and windows must be positive')
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    origin = max(r.info.start_us for r in readers)
    end = min(r.info.end_us for r in readers)
    margin = max(args.windows_us)
    if end - origin <= margin:
        raise RuntimeError('Common interval is shorter than the largest window')
    times = np.linspace(origin + margin, end - 1, args.samples, dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for window in args.windows_us:
        tiles = []
        for timestamp in times:
            panels = []
            for reader in readers:
                events = reader.window(int(timestamp - window), int(timestamp))
                panel = image(events, reader.info.width, reader.info.height, timestamp-window, timestamp)
                panels.append(cv2.cvtColor(cv2.resize(panel, (320, 180)), cv2.COLOR_GRAY2BGR))
            tile = np.hstack(panels)
            cv2.putText(tile, f't={(timestamp-origin)/1e6:.3f}s  window={window/1000:g}ms  M | S',
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        columns = 3
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0, len(tiles), columns)])
        out = args.output_dir / f'window_{window}us.jpg'
        if not cv2.imwrite(str(out), sheet):
            raise RuntimeError(f'Failed to write {out}')
        print(out.resolve(), flush=True)


if __name__ == '__main__':
    main()
