"""Qualitative held-out serve inference using each camera's final checkpoint."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from common import ROOT, DATA, EXP, dump, rows, table
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def predict_batched(model, paths, device, floor, observed):
    results = []
    handles = []
    # Ultralytics may copy/fuse the network during predictor setup. Hooks on
    # YOLO.model can disappear on later sequences. Observe the actual backend
    # after setup instead, and always remove callbacks/hooks between sequences.
    def attach(predictor):
        handles.append(predictor.model.register_forward_pre_hook(
            lambda module, inputs: observed.append(list(inputs[0].shape))))
    def detach(predictor=None):
        while handles:
            handles.pop().remove()
    model.add_callback('on_predict_start', attach)
    model.add_callback('on_predict_end', detach)
    # Explicit chunking: installed predictor batches file lists as one batch regardless of batch=.
    try:
        for i in range(0, len(paths), 16):
            chunk = paths[i:i+16]
            batch = model.predict(source=[str(p) for p in chunk], device=device, imgsz=256, batch=16,
                                  conf=floor, iou=0.7, rect=False, half=False, save=False, verbose=False)
            assert len(batch) == len(chunk)
            results.extend(batch)
    finally:
        detach()
        model.callbacks['on_predict_start'].remove(attach)
        model.callbacks['on_predict_end'].remove(detach)
    assert observed and all(s[1:] == [1, 256, 256] and s[0] <= 16 for s in observed)
    assert [Path(r.path).resolve() for r in results] == [p.resolve() for p in paths]
    return results


def boxes(result):
    return [dict(class_id=int(c), class_name='marker', confidence=float(s), xyxy_pixels=b.tolist(),
                 xywh_normalized=[float((b[0]+b[2])/512), float((b[1]+b[3])/512),
                                  float((b[2]-b[0])/256), float((b[3]-b[1])/256)])
            for b, s, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]


def draw_boxes(draw, candidates, offset, scale, font, color='cyan'):
    ox, oy = offset
    for candidate in candidates:
        x1, y1, x2, y2 = candidate['xyxy_pixels']
        draw.rectangle((ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale), outline=color, width=2)
        draw.text((ox+x1*scale, max(oy, oy+y1*scale-18)), f"marker {candidate['confidence']:.2f}",
                  font=font, fill=color, stroke_width=1, stroke_fill='black')


def panel(directory, row, candidates, confidence, font):
    image = Image.new('RGB', (1152, 456), '#161616')
    draw = ImageDraw.Draw(image)
    display = [b for b in candidates if b['confidence'] >= confidence]
    draw.text((5, 1), f"{row['group']} frame={row['frame_index']} timestamp={row['timestamp_us']} us", font=font, fill='white')
    support = row['track_support']
    draw.text((5, 23), f'UNLABELED | conf>={confidence:g} n={len(display)} | track={support}', font=font,
              fill='#ffcc00' if support == 'fallback_unverified' else 'cyan')
    filename = Path(row['image']).name
    for i, (sub, title) in enumerate([('binary', 'Filtered: model input'), ('binary_unfiltered', 'Unfiltered: same coordinates'),
                                     ('polarity_reference', 'Polarity: reference only')]):
        im = Image.open(directory / sub / filename).convert('RGB').resize((384, 384), Image.Resampling.NEAREST)
        image.paste(im, (i*384, 72))
        draw.text((i*384+5, 49), title, font=font, fill='white')
        draw_boxes(draw, display, (i*384, 72), 1.5, font)
    return image


def fit_check(camera, model, device, floor, confidence, observed):
    out = EXP / camera / 'training_fitcheck'
    out.mkdir()
    samples = json.loads((EXP / 'audit' / f'{camera}_GT_samples.json').read_text())
    paths = [DATA / 'datasets' / camera / 'images/train' / n for n in samples]
    results = predict_batched(model, paths, device, floor, observed)
    sheet = Image.new('RGB', (3*512, 4*544), '#161616')
    font = ImageFont.truetype(FONT_PATH, 16)
    for i, (path, result) in enumerate(zip(paths, results)):
        image = Image.new('RGB', (512, 544), '#161616')
        image.paste(Image.open(path).convert('RGB').resize((512,512), Image.Resampling.NEAREST), (0,32))
        draw = ImageDraw.Draw(image)
        draw.text((3,2), path.stem + ' | TRAIN ONLY', fill='white', font=font)
        label = DATA / 'datasets' / camera / 'labels/train' / (path.stem+'.txt')
        for line in label.read_text().splitlines():
            _, x,y,w,h=map(float,line.split())
            draw.rectangle(((x-w/2)*512,(y-h/2)*512+32,(x+w/2)*512,(y+h/2)*512+32), outline='lime', width=2)
        draw_boxes(draw, [b for b in boxes(result) if b['confidence'] >= confidence], (0,32), 2., font)
        image.save(out / (path.stem+'.jpg'), quality=90)
        sheet.paste(image, ((i%3)*512,(i//3)*544))
    sheet.save(out / 'contact_sheet_GT_green_prediction_cyan.jpg', quality=90)
    dump(out / 'README.json', dict(purpose='training fitting visualization only; not independent evaluation',
                                    gt_color='green', prediction_color='cyan', samples=samples,
                                    confidence=confidence, no_accuracy_metrics=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', required=True, choices=['F3E','F59'])
    parser.add_argument('--device', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--floor', type=float, default=0.05)
    args = parser.parse_args()
    assert 0 < args.floor <= args.conf < 1
    camera = args.camera
    completion = json.loads((EXP / camera / 'training_completion.json').read_text())
    assert completion['status'] == 'completed'
    checkpoint = EXP / camera / 'train/weights/last.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == completion['checkpoint_sha256']
    out = EXP / camera / 'qualitative'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    model = YOLO(str(checkpoint))
    assert model.model.model[0].conv.in_channels == 1 and model.names == {0:'marker'}
    observed = []
    font = ImageFont.truetype(FONT_PATH, 16)
    fit_check(camera, model, args.device, args.floor, args.conf, observed)
    summaries, all_records = [], []
    started = time.monotonic()
    for group_dir in sorted((DATA / 'unlabeled_inference' / camera).glob('serve*')):
        name = group_dir.name
        records = sorted(rows(group_dir / 'manifest.csv'), key=lambda r:int(r['timestamp_us']))
        metadata = json.loads((group_dir / 'metadata.json').read_text())
        group_out = out / name
        (group_out / 'overlays').mkdir(parents=True)
        video = DATA / metadata['reference_video']
        (group_out / 'reference_original.mp4').symlink_to(os.path.relpath(video, group_out))
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                           '-show_entries','stream=r_frame_rate,nb_frames','-of','json',str(video)],text=True))['streams'][0]
        fps_num,fps_den = map(int,probe['r_frame_rate'].split('/'))
        video_fps = fps_num/fps_den
        video_frames = int(probe['nb_frames'])
        paths = [group_dir / r['image'] for r in records]
        results = predict_batched(model, paths, args.device, args.floor, observed)
        group_records, detection_rows = [], []
        encoder_log = (group_out / 'video_encoding.log').open('wb')
        process = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                                    '-s','1152x456','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
                                    '-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',
                                    str(group_out/'review_slow.mp4')], stdin=subprocess.PIPE,stderr=encoder_log)
        try:
            for row,result in zip(records,results):
                candidates = boxes(result)
                visible = [b for b in candidates if b['confidence'] >= args.conf]
                frame_float = (int(row['timestamp_us'])-int(metadata['video_first_us']))/int(metadata['video_step_us'])
                video_index = int(math.floor(frame_float+0.5))
                reference_valid = 0 <= video_index < video_frames
                rec = dict(camera=camera, group=name, image=Path(row['image']).name, frame_index=int(row['frame_index']),
                           timestamp_us=int(row['timestamp_us']), window_us=int(row['window_us']), track_support=row['track_support'],
                           center_x_px=float(row['center_x_px']), center_y_px=float(row['center_y_px']), radius_px=float(row['radius_px']),
                           display_detections=len(visible), candidates_at_floor=len(candidates),
                           max_confidence=max((b['confidence'] for b in candidates), default=0.0),
                           reference_video=metadata['reference_video'], reference_video_frame_estimate=video_index,
                           reference_video_player_seconds=video_index/video_fps if reference_valid else None,
                           reference_timestamp_error_us=(int(metadata['video_first_us'])+video_index*int(metadata['video_step_us'])-int(row['timestamp_us'])),
                           reference_index_in_bounds=reference_valid, reference_timebase_status='metadata_estimate_unverified_not_exact_sync',
                           inference_ms=result.speed['inference'], detections=candidates)
                group_records.append(rec)
                all_records.append(rec)
                for i,b in enumerate(candidates):
                    detection_rows.append(dict(image=rec['image'], timestamp_us=rec['timestamp_us'], track_support=rec['track_support'],
                                               candidate_id=i, confidence=b['confidence'], class_name='marker',
                                               x1=b['xyxy_pixels'][0], y1=b['xyxy_pixels'][1], x2=b['xyxy_pixels'][2], y2=b['xyxy_pixels'][3],
                                               displayed=b['confidence'] >= args.conf))
                rendered = panel(group_dir,row,candidates,args.conf,font)
                rendered.save(group_out/'overlays'/(Path(row['image']).stem+'.jpg'),quality=88)
                process.stdin.write(rendered.tobytes())
        finally:
            process.stdin.close()
            code=process.wait()
            encoder_log.close()
        assert code == 0,(group_out/'video_encoding.log').read_text()
        probe_out=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                             '-show_entries','stream=nb_read_frames,codec_name,width,height','-of','json',str(group_out/'review_slow.mp4')],text=True))['streams'][0]
        assert int(probe_out['nb_read_frames']) == len(records)
        dump(group_out/'video_metadata.json',probe_out)
        dump(group_out/'predictions.json',group_records)
        table(group_out/'frames.csv',[{k:v for k,v in r.items() if k!='detections'} for r in group_records])
        table(group_out/'detections.csv',detection_rows,['image','timestamp_us','track_support','candidate_id','confidence','class_name','x1','y1','x2','y2','displayed'])
        table(group_out/'manual_review.csv',[dict(image=r['image'],timestamp_us=r['timestamp_us'],track_support=r['track_support'],
                                                crop_valid='',marker_visible='',missing_boxes='',extra_boxes='',notes='') for r in group_records])
        selected=np.linspace(0,len(records)-1,6,dtype=int).tolist()
        sheet=Image.new('RGB',(2304,1368),'#161616')
        for i,n in enumerate(selected):
            sheet.paste(Image.open(group_out/'overlays'/(Path(records[n]['image']).stem+'.jpg')),((i%2)*1152,(i//2)*456))
        sheet.save(group_out/'contact_sheet.jpg',quality=90)
        fallback=[r for r in group_records if r['track_support']=='fallback_unverified']
        supported=[r for r in group_records if r['track_support']!='fallback_unverified']
        def counts(rs):
            return dict(frames=len(rs),frames_with_predictions=sum(r['display_detections']>0 for r in rs),
                        frames_without_predictions=sum(r['display_detections']==0 for r in rs),detections=sum(r['display_detections'] for r in rs))
        summary=dict(camera=camera,group=name,**counts(group_records),fallback=counts(fallback),ring_supported=counts(supported),
                     reference_indices_out_of_bounds=sum(not r['reference_index_in_bounds'] for r in group_records),
                     inference_exceptions=0,verified_video_frames=int(probe_out['nb_read_frames']))
        dump(group_out/'summary.json',summary)
        summaries.append(summary)
        print('DONE',camera,name,summary,flush=True)
    dump(out/'predictions_all.json',all_records)
    table(out/'frames_all.csv',[{k:v for k,v in r.items() if k!='detections'} for r in all_records])
    table(out/'group_summary.csv',[{k:v for k,v in g.items() if k not in ('fallback','ring_supported')} for g in summaries])
    dump(out/'summary.json',dict(status='completed',camera=camera,groups=summaries,total_frames=len(all_records),
                                 frames_with_predictions=sum(r['display_detections']>0 for r in all_records),
                                 frames_without_predictions=sum(r['display_detections']==0 for r in all_records),
                                 total_detections=sum(r['display_detections'] for r in all_records),
                                 fallback_frames=sum(r['track_support']=='fallback_unverified' for r in all_records),
                                 mean_inference_ms=float(np.mean([r['inference_ms'] for r in all_records])),
                                 elapsed_seconds=time.monotonic()-started,input_shapes=sorted(set(map(tuple,observed))),
                                 checkpoint=str(checkpoint),checkpoint_sha256=completion['checkpoint_sha256'],
                                 conf=args.conf,prediction_floor=args.floor,nms_iou=0.7,half=False,imgsz=256,
                                 effective_max_batch=16,video_fps=25,video_slowdown=80,
                                 unavailable_metrics=['precision','recall','mAP'],
                                 reason='Other serves lack human ground truth; all outputs are qualitative only.'))
    print('INFERENCE COMPLETED',camera,flush=True)


if __name__=='__main__':
    main()
