"""Ball-normalized marker evidence comparison; no marker identity or spin labels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_ball_event_filters import stc_cut_trail_mask
from pingpong_event_pipeline.events import RawWindowReader


def surfaces(events, end_us, window_us, path_t, centers, radii, size=192):
    ts = events['timestamp'].astype(np.int64, copy=False)
    selection = (ts >= end_us-window_us) & (ts < end_us)
    ev, ts = events[selection], ts[selection]
    cx = np.interp(ts, path_t, centers[:, 0]); cy = np.interp(ts, path_t, centers[:, 1])
    radius = np.interp(ts, path_t, radii)
    nx = (ev['x'].astype(float)-cx) / radius
    ny = (ev['y'].astype(float)-cy) / radius
    inner = nx*nx + ny*ny < .78**2
    valid = nx*nx + ny*ny < 1
    u = np.rint((nx[valid]+1)*size/2).astype(np.int32)
    v = np.rint((ny[valid]+1)*size/2).astype(np.int32)
    pol = ev['polarity'][valid] != 0
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, pol = u[inside], v[inside], pol[inside]
    accum = np.zeros((2, size, size), np.float32)
    np.add.at(accum[0], (v[~pol], u[~pol]), 1)
    np.add.at(accum[1], (v[pol], u[pol]), 1)
    return accum, {'inside_ball': int(np.count_nonzero(valid)), 'inside_078': int(np.count_nonzero(inner))}


def color_counts(surface, mode, size=192):
    components = []
    for plane in surface:
        fine = cv2.GaussianBlur(plane, (0, 0), 1.1)
        if mode == 'RAW':
            value = np.clip(plane * 150, 0, 255)
        elif mode == 'DENSITY':
            value = np.clip(np.sqrt(fine) * 500, 0, 255)
        else:
            broad = cv2.GaussianBlur(plane, (0, 0), 7)
            value = np.clip(np.sqrt(np.maximum(fine - broad, 0)) * 600, 0, 255)
        components.append(value.astype(np.uint8))
    off, on = components
    image = np.zeros((size, size, 3), np.uint8)
    image[:, :, 0] = off
    image[:, :, 1] = on
    image[:, :, 2] = (on.astype(np.uint16)*.85).astype(np.uint8)
    yy, xx = np.ogrid[:size, :size]
    image[(xx-size/2)**2+(yy-size/2)**2 > (size*.39)**2] //= 4
    cv2.circle(image, (size//2, size//2), round(size*.39), (75, 75, 75), 1)
    return cv2.resize(image, (288, 288), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    track = np.load(args.path)
    path_t = track['timestamps_us'].astype(np.int64)
    centers = track['centers_px'].astype(float); radii = track['radii_px'].astype(float)
    reader = RawWindowReader(args.raw)
    if (len(path_t) < 4 or np.any(np.diff(path_t) <= 0) or np.any(~np.isfinite(centers))
            or np.any(~np.isfinite(radii)) or np.any(radii <= 0)):
        raise ValueError('invalid trajectory')
    if 'measured_scores' not in track or np.any(track['measured_scores'] <= 0):
        raise ValueError('trajectory contains undetected/extrapolated bins; review and restrict the interval first')
    if (np.any(centers[:, 0]-radii < 0) or np.any(centers[:, 1]-radii < 0)
            or np.any(centers[:, 0]+radii >= reader.info.width)
            or np.any(centers[:, 1]+radii >= reader.info.height)):
        raise ValueError('trajectory reaches image edge; full ball not visible')
    ev = reader.window(int(path_t[0])-15000, int(path_t[-1])+1)
    ev = ev[np.argsort(ev['timestamp'], kind='stable')]
    x = ev['x'].astype(np.int32); y = ev['y'].astype(np.int32)
    if np.any(x >= reader.info.width) or np.any(y >= reader.info.height):
        raise RuntimeError('out of bounds event coordinates')
    keep = stc_cut_trail_mask(x, y, ev['timestamp'].astype(np.int64),
        ev['polarity'].astype(np.int8), reader.info.width, reader.info.height, 2000)
    filtered = ev[keep]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_dir/'marker_comparison.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 25, (1152, 576))
    if not writer.isOpened():
        raise RuntimeError('video writer failed')
    details = []
    render_times = np.arange(path_t[0]+2000, path_t[-1], 2000, dtype=np.int64)
    checkpoints = {0, len(render_times)//2, len(render_times)-1}
    try:
        for i, timestamp in enumerate(render_times):
            row_images = []
            for window in (1000, 2000):
                raw, raw_count = surfaces(ev, timestamp, window, path_t, centers, radii)
                clean, clean_count = surfaces(filtered, timestamp, window, path_t, centers, radii)
                variants = [('RAW', raw, 'RAW'), ('DENSITY', raw, 'DENSITY'),
                            ('LOCAL', raw, 'LOCAL'), ('LOCAL+STC', clean, 'LOCAL')]
                images = []
                for label, surface, mode in variants:
                    tile = color_counts(surface, mode)
                    cv2.putText(tile, f'{label} {window/1000:g}ms', (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (255, 255, 255), 1, cv2.LINE_AA)
                    images.append(tile)
                row_images.append(np.hstack(images))
                details.append({'end_us': int(timestamp), 'window_us': window,
                                'raw': raw_count, 'stc': clean_count})
            view = np.vstack(row_images)
            writer.write(view)
            if i in checkpoints:
                cv2.imwrite(str(args.output_dir/f'comparison_{i:03d}.png'), view)
    finally:
        writer.release()
    summary = {'raw': str(Path(args.raw).resolve()), 'window_us': [1000, 2000], 'video_step_us': 2000,
        'video_fps': 25, 'inner_radius_fraction': .78, 'stc_threshold_us': 2000,
        'warnings': ['Display-only enhancement; no verified marker labels, identities or spin.',
                     'Ball centers and radius are provisional; events may include background.',
                     'STC algorithm is experimental, not a reproduction of hardware filter.'],
        'details': details}
    (args.output_dir/'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'output={args.output_dir.resolve()} frames={len(render_times)}', flush=True)


if __name__ == '__main__':
    main()
