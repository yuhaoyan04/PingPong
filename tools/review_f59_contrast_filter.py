"""Visualize F59 multi-scale local activity contrast without modifying data."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

from pingpong_event_pipeline.event_denoise import multiscale_contrast_filter, neighbor_filter
from pingpong_event_pipeline.events import RawWindowReader, trajectory_compensated_ball_patch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--pipeline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--crop-scale', type=float, default=1.2)
    p.add_argument('--moderate', type=float, default=.035)
    p.add_argument('--strong', type=float, default=.08)
    args = p.parse_args()
    if args.crop_scale < 1 or args.moderate < 0 or args.strong < args.moderate:
        raise ValueError('invalid scale or thresholds')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entries = json.loads((args.pipeline/'summary.json').read_text(encoding='utf8'))['groups']
    reader = None; reader_raw = None
    for entry in entries:
        name = entry['group']
        if not name.endswith('_F59'): continue
        track = args.pipeline/name/'raw_refined_path.npz'
        if not track.exists(): continue
        meta = json.loads((args.groups/name/'source.json').read_text(encoding='utf8'))
        if reader_raw != meta['raw']:
            reader = RawWindowReader(meta['raw']); reader_raw = meta['raw']
        records = np.load(track)['records']; path = records[:, :4]
        if len(records) < 2: continue
        events = reader.window(int(records[0,0])-1000, int(records[-1,0])+1)
        events = events[np.argsort(events['timestamp'], kind='stable')]
        timestamps = events['timestamp'].astype(np.int64, copy=False)
        group = args.output/name; group.mkdir()
        writer = cv2.VideoWriter(str(group/'full_frame_filter_comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920,400))
        if not writer.isOpened(): raise RuntimeError('cannot open output video')
        counts = []
        try:
            for i, row in enumerate(records):
                now = int(row[0]); lo=np.searchsorted(timestamps,now-1000); hi=np.searchsorted(timestamps,now)
                sample = events[lo:hi]
                valid = (sample['x'] < reader.info.width) & (sample['y'] < reader.info.height)
                sample = sample[valid]
                full = np.full((reader.info.height,reader.info.width,3),90,np.uint8)
                pol=sample['polarity'].astype(bool)
                full[sample['y'][~pol],sample['x'][~pol]]=(255,0,0)
                full[sample['y'][pol],sample['x'][pol]]=(0,255,255)
                full=cv2.resize(full,(640,360),interpolation=cv2.INTER_AREA)
                for j in range(max(1,i-100),i+1):
                    a,b=records[j-1],records[j]
                    if b[0]-a[0]==1000 and a[5] and b[5]:
                        cv2.line(full,tuple(np.rint(a[1:3]*.5).astype(int)),tuple(np.rint(b[1:3]*.5).astype(int)),(255,0,255),1)
                center=tuple(np.rint(row[1:3]*.5).astype(int))
                cv2.circle(full,center,max(1,round(row[3]*.5)),(255,0,255) if row[5] else (0,165,255),1)
                cv2.circle(full,center,max(1,round(row[3]*args.crop_scale*.5)),(255,255,255),1)
                surf=trajectory_compensated_ball_patch(sample,path[:,0].astype(np.int64),path[:,1:3],path[:,3]*args.crop_scale,256,now,500,1000)
                raw=(np.max(surf,axis=0)*255>45).astype(np.uint8)*255
                old=neighbor_filter(raw,3,2)
                moderate=multiscale_contrast_filter(raw,args.moderate)
                strong=multiscale_contrast_filter(raw,args.strong)
                removed=cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)
                removed[(raw!=0)&(strong==0)]=(0,0,255)
                canvas=np.zeros((400,1920,3),np.uint8); canvas[:360,:640]=full
                for k,img in enumerate((raw,old,moderate,strong,removed)):
                    panel=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if img.ndim==2 else img
                    canvas[28:284,640+k*256:640+(k+1)*256]=panel
                for k,title in enumerate(('RAW CROP','OLD SPATIAL','CONTRAST MEDIUM','CONTRAST STRONG','RED = REMOVED STRONG')):
                    cv2.putText(canvas,title,(645+k*256,19),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                status='RING SUPPORTED' if row[5] else 'FALLBACK - NOT DETECTED'
                cv2.putText(canvas,f'{name} {now} us {status} | purple: track, white: padded crop | MED {args.moderate:.3f} STR {args.strong:.3f}',
                            (8,385),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                writer.write(canvas)
                if i==len(records)//2: cv2.imwrite(str(group/'comparison_middle.png'),canvas)
                counts.append((now,int(bool(row[5])),int(np.count_nonzero(raw)),int(np.count_nonzero(old)),int(np.count_nonzero(moderate)),int(np.count_nonzero(strong))))
        finally: writer.release()
        with (group/'pixel_counts.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(('timestamp_us','ring_supported','raw_pixels','old_pixels','moderate_pixels','strong_pixels'));w.writerows(counts)
        print(name,len(records),flush=True)
    (args.output/'README.txt').write_text(
      f'Experimental F59 comparison: 1 ms windows and 1 ms steps; 25 fps playback. '
      f'Two-scale occupancy difference sigma=3,12 crop pixels; thresholds {args.moderate} and {args.strong}. '
      'Output pixels are a subset of RAW. Red means removed in strong variant. '
      'This is NOT a marker detector or ground truth. Dense edges/trails can pass and faint marks can be removed. '
      'All circles and fallback positions require review. Original RAW and annotations remain untouched.\n',encoding='utf8')


if __name__=='__main__': main()
