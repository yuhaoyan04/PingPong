"""Render RAW/reference comparisons and full-field tracking/spin review videos."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.raw_sync import IndexedEvents, load_serves, sequence_hash
from pingpong_event_pipeline.clip_sync import ffmpeg_writer, finish
from pingpong_event_pipeline.radius_review import CachedDraw, font, arrow
from pingpong_event_pipeline.motion_review import color
from pingpong_event_pipeline.motion import write_csv


def gray_frame(events):
    image = np.full((720, 1280), 127, np.uint8)
    image[events['y'], events['x']] = np.where(events['polarity'], 255, 0).astype(np.uint8)
    return image


def occupancy_iou(old, events):
    """Ignore original burned-in labels; compare independently decoded occupied pixels."""
    mask = np.abs(old.astype(np.int16) - 127) > 45
    mask[:40] = False
    e = events[events['y'] >= 40]
    ids = np.unique(e['y'].astype(np.int64) * 1280 + e['x'])
    hits = int(np.count_nonzero(mask.reshape(-1)[ids]))
    return hits / max(1, int(np.count_nonzero(mask)) + len(ids) - hits)


def build(dataset, output, cache, radius, tracks):
    cv2.setNumThreads(1)
    dataset, output, cache, radius = map(Path, [dataset, output, cache, radius])
    for part in ['full_view', 'tracking_spin', 'posters']:
        (output / part).mkdir(parents=True, exist_ok=True)
    overlays = {s['serve']: s for s in json.loads((radius / 'frame_overlays.json').read_text())['serves']}
    with Path(tracks).open(encoding='utf-8-sig') as stream:
        track_rows = list(csv.DictReader(stream))
    points = defaultdict(list); segments = defaultdict(list)
    for row in track_rows:
        points[(row['group'], int(row['timestamp_us']))].append(row)
        segments[(row['group'], row['segment_id'])].append(row)
    for rows in segments.values():
        rows.sort(key=lambda r: int(r['timestamp_us']))
    window_records = []; phase_records = []; summaries = []
    for spec in load_serves(dataset):
        name = spec['serve']
        sources = {c: IndexedEvents(np.load(cache / f'{name}_{c}.npy', mmap_mode='r')) for c in ['F3E', 'F59']}
        maps = {c: {int(r['timestamp_us']): r for r in spec['cameras'][c]['rows']} for c in sources}
        cap = cv2.VideoCapture(str(dataset / spec['cameras']['F3E']['metadata']['reference_video']))
        path = output / 'full_view' / f'{name}_raw_reference.mp4'
        process, log = ffmpeg_writer(path, 1280, 900, 30)
        frame_count = 0; per_camera = defaultdict(list)
        for i in range(spec['reference_frames']):
            ok, original = cap.read()
            if not ok:
                raise RuntimeError(f'{name}: reference decode failed at {i}')
            t = spec['reference_first_us'] + i * spec['reference_step_us']
            canvas = Image.new('RGB', (1280, 900), '#142130'); draw = CachedDraw(canvas)
            draw.text((12, 8), f'{name} | 两侧均取 [{t-1000}, {t}) μs | 源步长 5 ms', font=font(21), fill='white')
            draw.text((12, 43), '上排：RAW 顺序解码重建；下排：旧参考视频。同一时间标签，不平移任一相机。', font=font(19), fill='#95e5c3')
            for j, c in enumerate(sources):
                e = sources[c].window(t - 1000, t)
                panel = cv2.cvtColor(original[:, j*1280:(j+1)*1280], cv2.COLOR_BGR2GRAY)
                iou = occupancy_iou(panel, e); per_camera[c].append(iou)
                row = {'serve': name, 'reference_frame': i, 'camera': c, 'start_us': t-1000, 'end_us': t,
                       'events': len(e), 'first_event_us': int(e['timestamp'][0]) if len(e) else None,
                       'last_event_us': int(e['timestamp'][-1]) if len(e) else None,
                       'event_sequence_sha256': sequence_hash(e), 'old_reference_occupied_pixel_iou': iou}
                window_records.append(row)
                # Sparse phase search checks time labels against the image content.
                # This tests each export's RAW mapping, not cross-camera hardware sync.
                if i % 10 == 0 and 3 <= i < spec['reference_frames'] - 3:
                    candidates = [(occupancy_iou(panel, sources[c].window(t-1000+delta, t+delta)), delta)
                                  for delta in range(-10000, 10001, 1000)]
                    best_iou, shift = max(candidates)
                    phase_records.append({'serve': name, 'camera': c, 'reference_frame': i,
                                          'declared_end_us': t, 'best_raw_time_shift_us': shift,
                                          'best_iou': best_iou, 'zero_shift_iou': iou})
                gray = cv2.resize(gray_frame(e), (640, 360), interpolation=cv2.INTER_NEAREST)
                canvas.paste(Image.fromarray(gray).convert('RGB'), (j*640, 118))
                old = cv2.resize(original[:, j*1280:(j+1)*1280], (640, 360), interpolation=cv2.INTER_AREA)
                canvas.paste(Image.fromarray(cv2.cvtColor(old, cv2.COLOR_BGR2RGB)), (j*640, 518))
                draw.text((j*640+12, 83), f'{c} RAW | {len(e)} events | t={t}', font=font(17), fill='#ebd28e')
                draw.text((j*640+12, 486), f'{c} 旧参考帧 {i} | 事件像素 IoU={iou:.3f}', font=font(18), fill='#cddbe8')
            if i == spec['reference_frames']//2:
                canvas.save(output / 'posters' / f'{name}_raw_reference.png')
            process.stdin.write(np.asarray(canvas).tobytes()); frame_count += 1
        cap.release(); finish(process, log)
        print(name, 'RAW/reference rendered', frame_count, flush=True)
        overlay = overlays[name]
        process, log = ffmpeg_writer(output / 'tracking_spin' / f'{name}_raw_tracking_spin.mp4', 1280, 1030, 25)
        for index, t in enumerate(overlay['timestamps_us']):
            canvas = Image.new('RGB', (1280, 1030), '#142130'); draw = CachedDraw(canvas)
            draw.text((12, 7), f'{name} | RAW共同事件窗 [{t-1000}, {t}) μs | 采样 0.5 ms', font=font(21), fill='white')
            draw.text((12, 41), '完整视野始终重建；下方裁剪只显示已有导出。转速与方向为未标定的离线候选。', font=font(19), fill='#ebd28e')
            for j, c in enumerate(sources):
                e = sources[c].window(t-1000, t); x = j*640
                image = Image.fromarray(cv2.resize(gray_frame(e), (640, 360), interpolation=cv2.INTER_NEAREST)).convert('RGB')
                canvas.paste(image, (x, 114))
                draw.text((x+12, 81), f'{c} RAW | t={t} μs | {len(e)} events', font=font(17), fill='#95e5c3')
                row = maps[c].get(t)
                if row:
                    cx, cy, r = [float(row[k])*.5 for k in ['center_x_px','center_y_px','radius_px']]
                    draw.ellipse((x+cx-r,114+cy-r,x+cx+r,114+cy+r),outline='#ffcb60',width=2)
                    patch = Image.open(Path(spec['cameras'][c]['folder']) / row['image']).convert('RGB')
                    canvas.paste(patch, (x+20, 512))
                    draw.text((x+20, 481), f"已有裁剪源帧 {row['frame_index']} | {row['track_support']}", font=font(16), fill='#cddbe8')
                    group = f'{name}_{c}'
                    for marker in points.get((group, t), []):
                        col = color(marker['track_id'])
                        tail = [p for p in segments[(group, marker['segment_id'])] if 0 <= t-int(p['timestamp_us']) <= 4000]
                        xy = [(x+20+float(p['center_x_px']), 512+float(p['center_y_px'])) for p in tail]
                        if len(xy)>1: draw.line(xy, fill=col, width=2)
                        mx, my = x+20+float(marker['center_x_px']), 512+float(marker['center_y_px'])
                        draw.ellipse((mx-4,my-4,mx+4,my+4),outline=col,width=2)
                else:
                    draw.rectangle((x+20,512,x+276,768),fill='#080e16',outline='#506476')
                    draw.text((x+35,610), '此时刻未导出裁剪', font=font(20), fill='#ebd28e')
                value = overlay['values'][index][0]
                direction = value.get('directions',{}).get(c)
                arrow(draw, (x+410, 600), direction, '#95e5c3')
                draw.text((x+310, 659), f'{c} 相机坐标下的正面运动', font=font(17), fill='#cddbe8')
                draw.text((x+310, 689), '箭头不代表世界坐标上/下旋', font=font(16), fill='#ebd28e')
            draw.text((12, 794), '同一轨迹、同一拟合窗口；仅改变球面半径假设：', font=font(21), fill='white')
            for k, (factor, value) in enumerate(zip([1,1.1,1.2,1.5], overlay['values'][index])):
                label = f"{value['rps']:.2f} 转/秒" if value['available'] else 'N/A'
                draw.text((12+k*320, 836), f'R × {factor:g}：{label}', font=font(23), fill='#95e5c3' if value['available'] else '#ebd28e')
                draw.text((12+k*320, 879), str(value.get('source') or '无有效观测'), font=font(18), fill='#cddbe8')
            draw.text((12, 934), '方向箭头采用 R×1 候选；无当前观测时不延用旧值。25 FPS 慢放不用于计算事件时间。', font=font(18), fill='#cddbe8')
            draw.text((12, 976), '当前核验导出与 RAW 的时间对应；两相机物理同步精度仍需独立同步锚点。', font=font(18), fill='#ebd28e')
            if index == len(overlay['timestamps_us'])//2:
                canvas.save(output / 'posters' / f'{name}_raw_tracking_spin.png')
            process.stdin.write(np.asarray(canvas).tobytes())
        finish(process, log)
        summaries.append({'serve': name, 'reference_frames': frame_count, 'tracking_frames': len(overlay['timestamps_us']),
                          **{f'{c}_median_reference_iou': float(np.median(per_camera[c])) for c in sources}})
        print(name, 'tracking/spin rendered', len(overlay['timestamps_us']), flush=True)
    write_csv(output / 'reference_raw_frames.csv', window_records)
    write_csv(output / 'reference_phase_search.csv', phase_records)
    write_csv(output / 'video_summary.csv', summaries)
    report = {'serves': summaries, 'phase_samples': len(phase_records),
              'phase_search_step_us': 1000, 'phase_search_range_us': [-10000, 10000],
              'all_sampled_reference_phases_zero': all(r['best_raw_time_shift_us']==0 for r in phase_records),
              'two_camera_physical_sync_verified': False,
              'interpretation': 'Pixel evidence checks each export against its RAW; equal event-clock labels alone do not prove physical clock alignment.'}
    (output / 'render_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--radius', type=Path, default=Path('output/latest_radius'))
    p.add_argument('--tracks', type=Path, default=Path('output/latest/tracking/rotation_tracks_all.csv'))
    a = p.parse_args(); build(a.dataset, a.output, a.cache, a.radius, a.tracks)
