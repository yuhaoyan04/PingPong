"""Build a separate annotation workspace from supported RAW-track segments."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch
from pingpong_event_pipeline.event_denoise import neighbor_filter
from pingpong_event_pipeline.marker_enhancement import enhance_for_camera


def supported_runs(records, step_us=1000):
    indices = np.flatnonzero(records[:, 5] > 0)
    if not len(indices): return []
    breaks = (np.diff(indices) != 1) | (np.diff(records[indices, 0]) != step_us)
    return [records[run] for run in np.split(indices, np.flatnonzero(breaks)+1) if len(run) >= 3]


def candidate_runs(records, step_us=1000):
    """Keep fallbacks explicitly, but never interpolate across a timestamp gap."""
    arr = np.asarray(records, dtype=float).reshape(-1, 8)
    if len(arr) and np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError('trajectory timestamps must be strictly increasing')
    splits = np.flatnonzero(np.diff(arr[:, 0]) != step_us) + 1
    return [run for run in np.split(arr, splits) if len(run) >= 2]


def window_support(run, now):
    """A full 1 ms window is verified only if all bounding samples support it."""
    lo = np.searchsorted(run[:, 0], now-1000, side='right') - 1
    hi = np.searchsorted(run[:, 0], now, side='left')
    if lo < 0 or hi >= len(run): return False
    return bool(np.all(run[lo:hi+1, 5] > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', required=True, type=Path)
    parser.add_argument('--originals', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--crop-scale', type=float, default=1.0,
                        help='ROI margin relative to estimated radius, not a new radius measurement')
    parser.add_argument('--denoise', action='store_true', help='filter isolated occupied pixels; save the original too')
    parser.add_argument('--neighbor-radius', type=int, default=3)
    parser.add_argument('--min-neighbors', type=int, default=2)
    parser.add_argument('--camera-profiles', type=Path, help='JSON per-camera enhancement configuration')
    parser.add_argument('--include-fallback', action='store_true', help='also export explicit low-confidence fallback candidate crops')
    args = parser.parse_args()
    if args.crop_scale < 1.0: raise ValueError('crop-scale must be >= 1')
    if args.camera_profiles and args.denoise: raise ValueError('choose --denoise or --camera-profiles')
    profiles = json.loads(args.camera_profiles.read_text(encoding='utf8'))['cameras'] if args.camera_profiles else None
    if args.output.exists(): raise FileExistsError(f'refusing to overwrite {args.output}')
    summary = json.loads((args.pipeline/'summary.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    readers, results = {}, []
    for entry in summary['groups']:
        name = entry['group']; track = args.pipeline/name/'raw_refined_path.npz'
        if not track.is_file():
            results.append({'group': name, 'status': 'no_supported_track', 'frames': 0}); continue
        records = np.load(track)['records']
        runs = candidate_runs(records) if args.include_fallback else supported_runs(records)
        if not runs:
            results.append({'group': name, 'status': 'no_supported_run', 'frames': 0}); continue
        source = json.loads((args.originals/name/'source.json').read_text(encoding='utf-8'))
        camera = source['camera']
        if profiles is not None and camera not in profiles: raise ValueError(f'no profile for {camera}')
        if source['raw'] not in readers: readers[source['raw']] = RawWindowReader(source['raw'])
        reader = readers[source['raw']]
        events = reader.window(int(runs[0][0, 0])-1000, int(runs[-1][-1, 0])+1)
        events = events[np.argsort(events['timestamp'], kind='stable')]
        event_t = events['timestamp'].astype(np.int64, copy=False)
        group = args.output/name; (group/'binary').mkdir(parents=True); (group/'polarity_reference').mkdir()
        filtering = args.denoise or profiles is not None
        if filtering: (group/'binary_unfiltered').mkdir()
        rows = []
        for run in runs:
            trajectory = run[:, :4]
            for now in np.arange(int(run[0, 0])+1000, int(run[-1, 0])+1, 500, dtype=np.int64):
                lo, hi = np.searchsorted(event_t, now-1000, side='left'), np.searchsorted(event_t, now, side='left')
                surf = trajectory_compensated_ball_patch(events[lo:hi], trajectory[:, 0].astype(np.int64),
                          trajectory[:, 1:3], trajectory[:, 3] * args.crop_scale, 256, int(now), 500, 1000)
                off, on = surf[0]*255 > 45, surf[1]*255 > 45
                binary = ((off | on)*255).astype(np.uint8)
                original = binary
                if args.denoise:
                    binary = neighbor_filter(original, args.neighbor_radius, args.min_neighbors)
                elif profiles is not None:
                    binary = enhance_for_camera(original, camera, profiles)
                color = np.zeros((256, 256, 3), np.uint8); color[off, 0] = 255; color[on, 1:] = 255
                filename = f'{name}_{len(rows):05d}.png'
                if not cv2.imwrite(str(group/'binary'/filename), binary) or not cv2.imwrite(str(group/'polarity_reference'/filename), color):
                    raise RuntimeError(f'failed to write {filename}')
                if filtering and not cv2.imwrite(str(group/'binary_unfiltered'/filename), original):
                    raise RuntimeError(f'failed to write original {filename}')
                rows.append({'group': name, 'image': f'binary/{filename}',
                             'polarity_reference': f'polarity_reference/{filename}', 'frame_index': len(rows),
                             'timestamp_us': int(now), 'window_us': 1000,
                             'track_support': 'ring_supported' if window_support(run, now) else 'fallback_unverified',
                             'center_x_px': float(np.interp(now, trajectory[:, 0], trajectory[:, 1])),
                             'center_y_px': float(np.interp(now, trajectory[:, 0], trajectory[:, 2])),
                             'radius_px': float(np.interp(now, trajectory[:, 0], trajectory[:, 3]))})
        if rows:
            with (group/'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        meta = {k: source[k] for k in ('raw', 'video', 'camera', 'video_first_us', 'video_step_us')}
        meta.update(frames=len(rows), track_status=('includes_fallback_unverified' if args.include_fallback
                                                    else 'event_supported_candidate_unverified'),
                    crop_radius_scale=args.crop_scale,
                    denoise={'enabled': filtering, 'profile': profiles[camera] if profiles is not None else
                             {'method': 'neighbor', 'radius': args.neighbor_radius, 'min_neighbors': args.min_neighbors} if args.denoise else None,
                             'original_subdir': 'binary_unfiltered' if filtering else 'binary'},
                    fallback_frames=sum(r['track_support'] == 'fallback_unverified' for r in rows),
                    pipeline_diagnostic_flags=entry.get('diagnostic_flags', []),
                    track_start_us=int(records[0, 0]), track_end_us=int(records[-1, 0]),
                    frame_interval_us=500, window_us=1000,
                    method='coarse_motion_event_warp_local_ring_offline_smoothing',
                    annotations='new image coordinates; prior annotations deliberately not copied',
                    warning='All circles unverified. Fallback frames are low confidence. No interpolation across timestamp gaps; tracks do not cover full clips.')
        (group/'source.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append({'group': name, 'status': 'unverified_candidate', 'frames': len(rows),
                        'fallback_frames': sum(r['track_support'] == 'fallback_unverified' for r in rows),
                        'diagnostic_flags': entry.get('diagnostic_flags', [])})
        print(name, f'{len(rows)} frames', flush=True)
    (args.output/'batch_report.json').write_text(json.dumps({'results': results,
        'warning': 'No ground-truth labels were copied. Manual review required.'}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
