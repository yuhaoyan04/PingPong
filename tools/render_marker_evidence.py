"""Conservative marker-event visualization, not intensity or physical-marker reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from pingpong_event_pipeline.events import RawWindowReader, render_bgr, sae, trajectory_compensated_ball_patch
from pingpong_event_pipeline.marker_temporal import MarkerTracker
from pingpong_event_pipeline.marker_clusters import cluster_events, confirm_offline, associate_offline
from propose_physical_marker_centers import polarity_midpoints
from review_real_serve_markers import colored


def assign_persistent(previous, candidates, max_distance=24.0, min_streak=3):
    """Match only consecutive frames; missing observations are never extrapolated."""
    current = []
    assignments = {}
    if previous and candidates:
        old = np.array([[p['x'], p['y']] for p in previous])
        new = np.array([[p['x'], p['y']] for p in candidates])
        distances = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        for row, col in zip(rows, cols):
            if distances[row, col] <= max_distance:
                assignments[int(col)] = previous[int(row)]['streak'] + 1
    for index, p in enumerate(candidates):
        item = dict(p)
        item['streak'] = assignments.get(index, 1)
        item['show'] = item['streak'] >= min_streak
        current.append(item)
    return current


def evidence_image(image, selected, support_radius=16):
    """Keep observed polarity pixels near accepted hypotheses, never fill fitted ellipses."""
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    off = image[:, :, 0] > 45
    on = image[:, :, 1] > 45
    mask = np.zeros((height, width), bool)
    fits = []
    for item in selected:
        local = (xx-item['x'])**2 + (yy-item['y'])**2 <= support_radius**2
        pixels = local & (off | on)
        coords = np.column_stack(np.where(pixels))[:, ::-1].astype(np.float32)
        if len(coords) < 8 or np.count_nonzero(local & off) < 2 or np.count_nonzero(local & on) < 2:
            continue
        mask |= pixels
        if len(coords) >= 5:
            (x, y), (a, b), angle = cv2.fitEllipse(coords.reshape(-1, 1, 2))
            if min(a, b) >= 2 and max(a, b) <= 2*support_radius and max(a, b)/min(a, b) <= 4:
                fits.append({'center': [float(x), float(y)], 'axes': [float(a), float(b)],
                             'angle_deg': float(angle), 'support_pixels': int(len(coords))})
    result = np.zeros_like(image)
    result[mask] = image[mask]
    return result, mask, fits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--window-us', type=int, default=1000)
    p.add_argument('--step-us', type=int, default=2000)
    p.add_argument('--min-streak', type=int, default=3)
    p.add_argument('--max-distance', type=float, default=24.0)
    p.add_argument('--support-radius', type=int, default=16)
    p.add_argument('--temporal', action='store_true', help='track candidates and label bounded predictions')
    p.add_argument('--max-gap', type=int, default=3, help='maximum consecutive predicted updates')
    p.add_argument('--clusters', action='store_true', help='polarity-agnostic cluster candidates and denoising review')
    p.add_argument('--neighbor-radius', type=int, default=4)
    p.add_argument('--min-neighbors', type=int, default=2)
    p.add_argument('--merge-radius', type=int, default=6)
    p.add_argument('--min-cluster-pixels', type=int, default=3)
    p.add_argument('--max-cluster-extent', type=int, default=None)
    p.add_argument('--max-cluster-aspect', type=float, default=None)
    p.add_argument('--observed-only', action='store_true', help='show event-supported regions only, no track or fit overlays')
    p.add_argument('--offline-confirm', action='store_true', help='confirm candidates with previous OR next time window')
    p.add_argument('--offline-tracks', action='store_true', help='associate strong and weak clusters over multiple windows')
    p.add_argument('--track-gap', type=int, default=1, help='maximum missing intermediate windows')
    p.add_argument('--track-min-observations', type=int, default=3)
    p.add_argument('--track-min-strong', type=int, default=1)
    p.add_argument('--confirm-distance', type=float, default=20)
    p.add_argument('--confirm-size-ratio', type=float, default=4)
    args = p.parse_args()
    if (args.offline_confirm or args.offline_tracks) and not args.clusters:
        p.error('offline modes require --clusters')
    if args.offline_confirm and args.offline_tracks:
        p.error('choose one offline mode')
    cv2.setNumThreads(1)
    track = np.load(args.path)
    t = track['timestamps_us'].astype(np.int64)
    centers = track['centers_px'].astype(float); radii = track['radii_px'].astype(float)
    if (args.window_us <= 0 or args.step_us <= 0 or args.min_streak <= 0 or args.max_distance <= 0
            or args.support_radius <= 0 or np.any(track['measured_scores'] <= 0)):
        raise ValueError('positive timing and fully observed ball track required')
    reader = RawWindowReader(args.raw)
    events = reader.window(int(t[0])-args.window_us, int(t[-1])+1)
    events = events[np.argsort(events['timestamp'], kind='stable')]
    ev_t = events['timestamp'].astype(np.int64, copy=False)
    times = np.arange(t[0]+args.window_us, t[-1], args.step_us, dtype=np.int64)
    def make_candidates(raw):
        if args.offline_tracks:
            result = cluster_events(raw, args.neighbor_radius, 2, args.merge_radius, 3, 12, 75, 5.0)
            for item in result[0]:
                item['strong'] = (item['pixels'] >= args.min_cluster_pixels
                                  and (args.max_cluster_extent is None or max(item['width'], item['height']) <= args.max_cluster_extent)
                                  and (args.max_cluster_aspect is None or max(item['width'], item['height']) / min(item['width'], item['height']) <= args.max_cluster_aspect))
            return result
        return cluster_events(raw, args.neighbor_radius, args.min_neighbors, args.merge_radius,
                              args.min_cluster_pixels, 6, args.max_cluster_extent, args.max_cluster_aspect)

    def compute_clusters(now):
        lo = np.searchsorted(ev_t, now-args.window_us, side='left')
        hi = np.searchsorted(ev_t, now, side='left')
        surface = trajectory_compensated_ball_patch(events[lo:hi], t, centers, radii, 256, int(now),
                                                    args.window_us/2, args.window_us)
        raw = colored(surface)
        result = make_candidates(raw)
        return raw, result

    if args.offline_confirm or args.offline_tracks:
        # Only retain proposal metadata; rebuild each frame for rendering to limit memory.
        offline_proposals = [compute_clusters(now)[1][0] for now in times]
        confirmed = (associate_offline(offline_proposals, args.confirm_distance, args.confirm_size_ratio,
                                      args.track_gap, args.track_min_observations, args.track_min_strong)
                     if args.offline_tracks else confirm_offline(offline_proposals, args.confirm_distance, args.confirm_size_ratio))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canvas_width = 1840 if args.clusters and not args.observed_only else 1540
    writer = cv2.VideoWriter(str(args.output_dir/'marker_evidence_review.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 25, (canvas_width, 400))
    if not writer.isOpened():
        raise RuntimeError('video writer unavailable')
    previous = []
    tracker = MarkerTracker(gate_px=args.max_distance, max_gap=args.max_gap, min_hits=args.min_streak) if args.temporal else None
    rows = []
    try:
        for index, now in enumerate(times):
            lo = np.searchsorted(ev_t, now-args.window_us, side='left')
            hi = np.searchsorted(ev_t, now, side='left')
            current = events[lo:hi]
            c = np.array([np.interp(now, t, centers[:, j]) for j in (0, 1)])
            r = float(np.interp(now, t, radii))
            full = cv2.resize(render_bgr(sae(current, reader.info.width, reader.info.height,
                                            int(now), args.window_us/2)), (640, 360), interpolation=cv2.INTER_AREA)
            cv2.circle(full, tuple(np.rint(c*.5).astype(int)), max(1, round(r*.5)), (0, 255, 0), 1)
            path = np.rint(centers*.5).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(full, [path], False, (0, 150, 0), 1)
            surface = trajectory_compensated_ball_patch(current, t, centers, radii, 256, int(now),
                                                        args.window_us/2, args.window_us)
            raw = colored(surface)
            if args.clusters:
                proposals, denoised, cluster_labels, cluster_kept = make_candidates(raw)
            else:
                proposals = polarity_midpoints(raw, 35, 40, .36, 6)
                proposals = [a for a in proposals if a['assigned_active_pixels'] >= 4
                             and 1.5 <= a['polarity_separation_px'] <= 40]
            tracked = tracker.update(proposals) if tracker is not None else []
            if args.offline_confirm or args.offline_tracks:
                selected = [p for j, p in enumerate(proposals) if j in confirmed[index]][:6]
            elif tracker is not None:
                selected = [{'x': s.x, 'y': s.y} for s in tracked if s.status == 'observed']
            else:
                previous = assign_persistent(previous, proposals, args.max_distance, args.min_streak)
                selected = [a for a in previous if a['show']]
            if args.clusters:
                chosen = [p for p in proposals if any(abs(p['x']-s['x']) < .001 and abs(p['y']-s['y']) < .001 for s in selected)]
                mask = cluster_kept & np.isin(cluster_labels, [p['label'] for p in chosen])
                evidence = np.zeros_like(raw)
                evidence[mask] = raw[mask]
                fits = []
                for item in chosen:
                    yy, xx = np.where(cluster_kept & (cluster_labels == item['label']))
                    coords = np.column_stack((xx, yy)).astype(np.float32)
                    if len(coords) >= 5:
                        (x, y), (a, b), angle = cv2.fitEllipse(coords.reshape(-1, 1, 2))
                        if min(a, b) >= 2 and max(a, b) <= 80 and max(a, b)/min(a, b) <= 4:
                            fits.append({'center': [float(x), float(y)], 'axes': [float(a), float(b)],
                                         'angle_deg': float(angle), 'support_pixels': int(len(coords))})
            else:
                evidence, mask, fits = evidence_image(raw, selected, args.support_radius)
            fit_view = evidence.copy()
            for fit in ([] if args.observed_only else fits):
                ellipse = (tuple(fit['center']), tuple(fit['axes']), fit['angle_deg'])
                cv2.ellipse(fit_view, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
            for state in ([] if args.observed_only else tracked):
                if state.status == 'observed':
                    point = (round(state.x), round(state.y))
                    cv2.putText(fit_view, f'T{state.id}', (point[0]+8, point[1]-7),
                                cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 255, 0), 1, cv2.LINE_AA)
                    continue
                point = (round(state.x), round(state.y))
                # Predicted position is an overlay only: do not fabricate event pixels.
                cv2.drawMarker(fit_view, point, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 13, 1)
                cv2.circle(fit_view, point, 10, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(fit_view, f'P{state.id}', (point[0]+8, point[1]-7),
                            cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 165, 255), 1, cv2.LINE_AA)
            if not np.any(mask):
                cv2.putText(evidence, 'NO QUALIFYING CANDIDATE', (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (200, 200, 200), 1, cv2.LINE_AA)
                if args.observed_only or not tracked:
                    cv2.putText(fit_view, 'NO QUALIFYING CANDIDATE', (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (200, 200, 200), 1, cv2.LINE_AA)
            tile_sources = ((raw, denoised, evidence) if args.observed_only else (raw, denoised, evidence, fit_view)) if args.clusters else (raw, evidence, fit_view)
            tiles = [cv2.resize(x, (300, 300), interpolation=cv2.INTER_NEAREST) for x in tile_sources]
            frame = np.zeros((400, canvas_width, 3), np.uint8)
            frame[:360, :640] = full
            for col, image in enumerate(tiles):
                frame[30:330, 640+300*col:940+300*col] = image
            titles = ([(8, 'TRACK'), (646, 'BALL EVENTS'), (946, 'DENOISED'), (1246, 'SELECTED CLUSTERS')] + ([] if args.observed_only else [(1546, 'TRACK HYPOTHESIS')])
                      if args.clusters else [(8, 'TRACK'), (646, 'BALL EVENTS'), (946, 'SELECTED EVIDENCE'), (1246, 'ELLIPSE HYPOTHESIS')])
            for x, label in titles:
                cv2.putText(frame, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            label = ('CURRENT EVENT-SUPPORTED CANDIDATES ONLY' if args.observed_only else
                     ('GREEN=CANDIDATE FIT ORANGE=PREDICTED' if args.temporal else 'GREEN IS MODEL NOT TRUTH'))
            cv2.putText(frame, f't={(now-t[0])/1000:.1f} ms | BLUE OFF / YELLOW ON | {label}',
                        (8, 383), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if index in {0, len(times)//2, len(times)-1}:
                cv2.imwrite(str(args.output_dir/f'review_{index:03d}.png'), frame)
            rows.append({'index': index, 'timestamp_us': int(now), 'proposals': len(proposals),
                         'temporally_stable': len(selected), 'kept_pixels': int(mask.sum()), 'ellipse_fits': fits,
                         'raw_active_pixels': int(np.count_nonzero(np.max(raw, axis=2) > 45)),
                         'denoised_pixels': int(np.count_nonzero(np.max(denoised, axis=2) > 45)) if args.clusters else None,
                         'tracks': [vars(s).copy() for s in tracked]})
    finally:
        writer.release()
    report = {'raw': str(Path(args.raw).resolve()), 'path': str(args.path.resolve()),
              'window_us': args.window_us, 'step_us': args.step_us, 'min_streak': args.min_streak,
              'max_distance_patch_px': args.max_distance, 'support_radius_patch_px': args.support_radius,
              'temporal_tracking': args.temporal, 'maximum_predicted_updates': args.max_gap, 'frames': rows,
              'candidate_method': 'polarity_agnostic_clusters' if args.clusters else 'paired_polarity',
              'cluster_parameters': {'neighbor_radius': args.neighbor_radius, 'min_neighbors': 2 if args.offline_tracks else args.min_neighbors,
                                     'merge_radius': args.merge_radius, 'min_cluster_pixels': args.min_cluster_pixels,
                                     'max_cluster_extent': args.max_cluster_extent, 'max_cluster_aspect': args.max_cluster_aspect},
              'observed_only_display': args.observed_only,
              'offline_confirmation': {'enabled': args.offline_confirm, 'distance_patch_px': args.confirm_distance,
                                       'size_ratio': args.confirm_size_ratio, 'neighbor_steps': 1},
              'offline_tracks': {'enabled': args.offline_tracks, 'max_gap_updates': args.track_gap,
                                 'minimum_observations': args.track_min_observations, 'minimum_strong': args.track_min_strong,
                                 'weak_neighbor_count': 2, 'weak_min_pixels': 3, 'weak_max_extent': 75,
                                 'weak_max_aspect': 5.0, 'weak_proposal_limit': 12, 'display_limit': 6},
              'warnings': ['No ground-truth marker masks or spin labels; sustained artifacts can pass the gate.',
                           'Blank means no qualifying candidate, not proof no marker is present.',
                           'Track IDs are temporary associations, not verified physical marker identities.',
                           'Predicted positions are extrapolations and do not represent observed event pixels.',
                           'Green ellipse is an event-cloud model, not a recovered physical dot.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'frames={len(rows)} displayed={sum(r["kept_pixels"]>0 for r in rows)} output={args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
