"""Full RAW trajectory and spatial/temporal candidate filters side by side."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from pingpong_event_pipeline.event_denoise import neighbor_filter, temporal_neighbor_filter
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def crop(events, path, now, scale):
    surface = trajectory_compensated_ball_patch(events, path[:, 0].astype(np.int64), path[:, 1:3],
                 path[:, 3]*scale, 256, now, 500, 1000)
    return (np.max(surface, axis=0)*255 > 45).astype(np.uint8)*255


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--pipeline', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--crop-scale', type=float, default=1.2)
    args = p.parse_args()
    if args.crop_scale < 1: raise ValueError('crop-scale must be >= 1')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    groups = json.loads((args.pipeline/'summary.json').read_text(encoding='utf8'))['groups']
    readers = {}
    for item in groups:
        name = item['group']; track = args.pipeline/name/'raw_refined_path.npz'
        old = args.baseline/name/'raw_refined_path.npz'
        if not track.exists() or not old.exists(): continue
        meta = json.loads((args.groups/name/'source.json').read_text(encoding='utf8'))
        if meta['raw'] not in readers: readers[meta['raw']] = RawWindowReader(meta['raw'])
        reader = readers[meta['raw']]
        records = np.load(track)['records']; baseline = np.load(old)['records']; path = records[:, :4]
        if len(records) < 2: continue
        old_by_time = {int(row[0]): row for row in baseline}
        events = reader.window(int(records[0, 0])-2000, int(records[-1, 0])+1)
        events = events[np.argsort(events['timestamp'], kind='stable')]
        times = events['timestamp'].astype(np.int64, copy=False)
        output = args.output/name; output.mkdir()
        writer = cv2.VideoWriter(str(output/'full_trajectory_and_denoise.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1408,400))
        if not writer.isOpened(): raise RuntimeError('cannot open video writer')
        try:
            previous = None
            for i, row in enumerate(records):
                now = int(row[0]); lo = np.searchsorted(times, now-1000); hi = np.searchsorted(times, now)
                sample = events[lo:hi]
                frame = np.full((reader.info.height, reader.info.width, 3), 90, np.uint8)
                valid = (sample['x'] < reader.info.width) & (sample['y'] < reader.info.height)
                sample = sample[valid]; pol = sample['polarity'].astype(bool)
                frame[sample['y'][~pol], sample['x'][~pol]] = (255, 0, 0)
                frame[sample['y'][pol], sample['x'][pol]] = (0, 255, 255)
                frame = cv2.resize(frame, (640,360), interpolation=cv2.INTER_AREA)
                for j in range(max(1,i-100), i+1):
                    a,b = records[j-1], records[j]
                    if b[0]-a[0] == 1000 and a[5] and b[5]:
                        cv2.line(frame, tuple(np.rint(a[1:3]*.5).astype(int)), tuple(np.rint(b[1:3]*.5).astype(int)), (255,0,255), 1)
                old_row = old_by_time.get(now)
                if old_row is not None:
                    cv2.circle(frame, tuple(np.rint(old_row[1:3]*.5).astype(int)), max(1,round(old_row[3]*.5)), (0,255,0), 1)
                old_status = 'OLD SUPPORTED' if old_row is not None and old_row[5] else 'OLD FALLBACK'
                color = (255,0,255) if row[5] else (0,165,255)
                cv2.circle(frame, tuple(np.rint(row[1:3]*.5).astype(int)), max(1,round(row[3]*.5)), color, 1)
                cv2.circle(frame, tuple(np.rint(row[1:3]*.5).astype(int)), max(1,round(row[3]*args.crop_scale*.5)), (255,255,255), 1)
                raw = crop(sample, path, now, args.crop_scale)
                spatial = neighbor_filter(raw, radius=3, min_neighbors=2)
                consecutive = i > 0 and records[i-1,0] == now-1000 and bool(row[5]) and bool(records[i-1,5])
                temporal = temporal_neighbor_filter(raw, previous, 3, 2, 6) if consecutive and previous is not None else np.zeros_like(raw)
                canvas = np.zeros((400,1408,3), np.uint8); canvas[:360,:640] = frame
                for k,img in enumerate((raw,spatial,temporal)):
                    canvas[28:284,640+k*256:640+(k+1)*256] = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                for k,title in enumerate(('PADDED RAW CROP','SPATIAL FILTER','TEMPORAL + SPATIAL')):
                    cv2.putText(canvas,title,(645+k*256,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                if not consecutive:
                    cv2.putText(canvas,'NO CONSECUTIVE SUPPORT',(1159,300),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,165,255),1)
                status = 'RING SUPPORTED' if row[5] else 'FALLBACK - NOT A DETECTION'
                cv2.putText(canvas,f'{name} {now}us {status} | {old_status} | green=old magenta=new white=ROI',
                            (8,385),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
                writer.write(canvas); previous = raw
                if i == len(records)//2: cv2.imwrite(str(output/'comparison_middle.png'),canvas)
        finally: writer.release()
        print(name,len(records),flush=True)
    (args.output/'README.txt').write_text('Green=old circle; magenta=adaptive supported circle and history; orange=fallback; white=crop ROI. '
      'Temporal filter requires two supported consecutive windows. Newly visible or rotating marks can be lost. '
      'All circles and filters require manual verification; support is not accuracy.\n',encoding='utf8')


if __name__ == '__main__': main()
