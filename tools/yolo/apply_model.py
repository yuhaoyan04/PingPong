"""Apply camera models with conservative temporal tracking, or replay frozen detections for comparisons."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from common import ROOT, DATA, EXP, dump, rows, table
from marker_tracker import BallGeometry, MarkerTracker, TrackerConfig
from track_segments import select_consecutive_segments, write_rotation_exports, ROTATION_FIELDS
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def natural_key(path):
    return [int(v) if v.isdigit() else v for v in re.split(r'(\d+)',path.name)]


def input_sequence(source,camera,input_fps):
    source=source.resolve()
    directory=source.parent if source.is_file() else source
    if directory.name=='binary':
        directory=directory.parent
    manifest=directory/'manifest.csv'
    metadata=json.loads((directory/'metadata.json').read_text()) if (directory/'metadata.json').exists() else {}
    if manifest.exists():
        records=sorted(rows(manifest),key=lambda r:int(r['timestamp_us']))
        if source.is_file():
            records=[r for r in records if Path(r['image']).name==source.name]
        paths=[directory/r['image'] for r in records]
        timebase='manifest_timestamp_us'
    else:
        paths=[source] if source.is_file() else sorted(directory.glob('*.png'),key=natural_key)
        if not paths and (directory/'binary').exists():
            paths=sorted((directory/'binary').glob('*.png'),key=natural_key)
        records=[dict(group=directory.name,image=str(p),frame_index=i,
                      timestamp_us=int(round(i*1e6/input_fps)),track_support='unknown') for i,p in enumerate(paths)]
        timebase=f'assumed_{input_fps:g}_fps_relative_time'
    if not paths:
        raise ValueError(f'No input frames: {source}')
    if any(not p.is_file() for p in paths):
        raise ValueError('Missing manifest image')
    for r in records:
        if r.get('group','').endswith(('_F3E','_F59')) and not r['group'].endswith('_'+camera):
            raise ValueError('Camera and source group mismatch')
    return directory,metadata,records,paths,timebase


def paint(image,items,color,font,offset,tracked=False):
    draw=ImageDraw.Draw(image)
    ox,oy=offset
    for item in items:
        b=item['xyxy_pixels']
        selected=color
        if tracked:
            selected={'optical_flow':'#ff9900','prediction_hold':'#ffff00'}.get(item['source'],'#00ff80')
        draw.rectangle((ox+b[0]*1.5,oy+b[1]*1.5,ox+b[2]*1.5,oy+b[3]*1.5),outline=selected,width=2)
        if tracked:
            label=f"ID {item['track_id']} {item.get('segment_id','').rsplit('/',1)[-1]} {item['source']}"
            trail=item.get('trail',[])
            if len(trail)>1:
                draw.line([(ox+x*1.5,oy+y*1.5) for x,y in trail],fill=selected,width=2)
        else:
            label=f"{item['confidence']:.2f}"
        draw.text((ox+b[0]*1.5,max(oy,oy+b[1]*1.5-14)),label,fill=selected,font=font,
                  stroke_width=1,stroke_fill='black')


def render(directory,row,path,original,candidates,processed,config,font,geometry):
    canvas=Image.new('RGB',(1536,456),'#151515')
    ref=directory/'polarity_reference'/path.name
    base=Image.fromarray(original).convert('RGB').resize((384,384),Image.Resampling.NEAREST)
    for i in range(4):
        pic=Image.open(ref).convert('RGB').resize((384,384),Image.Resampling.NEAREST) if i==3 and ref.exists() else base
        canvas.paste(pic,(i*384,72))
    draw=ImageDraw.Draw(canvas)
    baseline=[d for d in candidates if d['confidence']>=.25]
    strict=processed['high_confidence_after_filters']
    titles=[f'Before conf .25: n={len(baseline)}',f'High {config.birth_confidence:g} + ball ROI: n={len(strict)}',
            f'Eligible >= {config.minimum_consecutive_frames} frames: n={len(processed["tracks"])}','Polarity / reference + segment IDs']
    draw.text((6,2),f"{row.get('group',directory.name)} frame={row['frame_index']} t={row['timestamp_us']} us | predictions, no ground truth",font=font,fill='white')
    draw.text((6,25),'Green: detector  Orange: optical flow  Yellow: short prediction hold  |  circle: safe ROI',font=font,fill='#cccccc')
    for i,title in enumerate(titles):
        draw.text((i*384+4,49),title,font=font,fill='white')
    for i in (1,2,3):
        cx,cy=geometry.center_x*1.5+i*384,geometry.center_y*1.5+72
        r=(geometry.radius-config.edge_margin_pixels)*1.5
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline='#6666ff',width=1)
    paint(canvas,baseline,'cyan',font,(0,72))
    paint(canvas,strict,'cyan',font,(384,72))
    paint(canvas,processed['tracks'],'#00ff80',font,(768,72),tracked=True)
    paint(canvas,processed['tracks'],'#00ff80',font,(1152,72),tracked=True)
    return canvas


def run_sequence(camera,source,out,config,args,model=None):
    directory,meta,records,paths,timebase=input_sequence(source,camera,args.input_fps)
    out.mkdir(parents=True,exist_ok=False)
    (out/'overlays').mkdir()
    checkpoint=EXP/camera/'train/weights/last.pt'
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected=json.loads((EXP/camera/'training_completion.json').read_text())['checkpoint_sha256']
    if checkpoint_hash!=expected:
        raise ValueError('Camera checkpoint changed')
    provenance={}
    if args.replay_cache:
        cached=EXP/camera/'qualitative'/directory.name/'predictions.json'
        cached_summary=json.loads((EXP/camera/'qualitative/summary.json').read_text())
        if cached_summary['checkpoint_sha256']!=checkpoint_hash or cached_summary['prediction_floor']>config.continuation_confidence:
            raise ValueError('Cached detector settings incompatible with this tracker')
        old=json.loads(cached.read_text())
        by_name={r['image']:r for r in old}
        candidates=[]
        for record,path in zip(records,paths):
            entry=by_name[path.name]
            if int(record['timestamp_us'])!=entry['timestamp_us']:
                raise ValueError('Cached timestamp mismatch')
            candidates.append(entry['detections'])
        provenance=dict(detector='replayed unchanged frozen model outputs; postprocessing remains causal',
                        cached_predictions_sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
                        cached_predictions=str(cached),detector_input_channels=1)
    else:
        from infer_camera import predict_batched,boxes
        observed=[]
        predicted=predict_batched(model,paths,args.device,.05,observed)
        candidates=[boxes(r) for r in predicted]
        provenance=dict(detector='fresh model inference',input_shapes=sorted(set(map(tuple,observed))),detector_input_channels=1)
    if len(records)==1:
        config=replace(config,confirmation_hits=1)
    geometry=BallGeometry(center_x=args.center_x,center_y=args.center_y,
                          radius=args.ball_radius if args.ball_radius else 128/float(meta.get('crop_radius_scale',1.2)),
                          source='explicit_cli_geometry' if args.ball_radius else 'nominal_crop_geometry_not_independent_ball_fit')
    tracker=MarkerTracker(config,geometry)
    dump(out/'config.json',dict(camera=camera,source=str(source),model=str(checkpoint),checkpoint_sha256=checkpoint_hash,
                                tracker=asdict(config),geometry=asdict(geometry),timebase=timebase,
                                temporal_tracking_enabled=len(records)>1,causal=False,tracker_causal=True,
                                export_selection='offline contiguous evidence-backed segments; initial samples buffered for minimum-length confirmation',
                                nms_iou_at_detector=.7,
                                nms_iou_after_detector=config.duplicate_nms_iou,**provenance))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    output_records=[]; flat_tracks=[]; totals=Counter(); id_stats={}
    for row,path,ds in zip(records,paths,candidates):
        with Image.open(path) as im:
            if im.mode!='L' or im.size!=(256,256):
                raise ValueError('Input must match single-channel training crops, 256x256')
            image=np.asarray(im).copy()
        result=tracker.update(image,ds,int(row['timestamp_us']))
        raw=sum(d['confidence']>=.25 for d in ds)
        high_before_edge=sum(d['confidence']>=config.birth_confidence for d in ds)
        totals['frames']+=1; totals['baseline_boxes_025']+=raw
        totals['high_threshold_boxes_before_edge']+=high_before_edge
        totals['high_threshold_boxes_after_filters']+=len(result['high_confidence_after_filters'])
        totals['raw_stable_track_boxes']+=len(result['tracks'])
        totals.update('rejected_'+d['reason'] for d in result['rejected_detections'])
        output_records.append(dict(camera=camera,group=row.get('group',directory.name),image=path.name,
                                   frame_index=int(row['frame_index']),track_support=row.get('track_support','unknown'),
                                   baseline_boxes_025=raw,high_threshold_boxes_before_edge=high_before_edge,**result))
    raw_frames=output_records
    output_records,segments,removed,selection=select_consecutive_segments(
        raw_frames,config.minimum_consecutive_frames,config.consecutive_max_gap_us)
    dump(out/'raw_frames.json',raw_frames)
    write_rotation_exports(out,output_records,segments,removed,selection)
    with (out/'encoding.log').open('wb') as log:
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                                  '-s','1536x456','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
                                  '-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(out/'comparison_slow.mp4')],
                                 stdin=subprocess.PIPE,stderr=log)
        try:
            for row,path,ds,event in zip(records,paths,candidates,output_records):
                with Image.open(path) as im:
                    if im.mode!='L' or im.size!=(256,256):
                        raise ValueError('Input must match single-channel training crops, 256x256')
                    image=np.asarray(im).copy()
                totals['stable_track_boxes']+=len(event['tracks'])
                totals['frames_with_stable_tracks']+=bool(event['tracks'])
                totals.update(t['source'] for t in event['tracks'])
                for t in event['tracks']:
                    b=t['xyxy_pixels']
                    flat_tracks.append(dict(camera=camera,group=event['group'],image=path.name,
                                             frame_index=event['frame_index'],timestamp_us=int(row['timestamp_us']),
                                             segment_id=t['segment_id'],segment_length=t['segment_length'],
                                             segment_frame_offset=t['segment_frame_offset'],
                                             track_id=t['track_id'],source=t['source'],
                                             detection_confirmed_this_frame=t['detection_confirmed_this_frame'],
                                             last_detection_confidence=t['last_detection_confidence'],
                                             frames_since_detection=t['frames_since_detection'],unobserved_frames=t['unobserved_frames'],
                                             x1=b[0],y1=b[1],x2=b[2],y2=b[3]))
                    stat=id_stats.setdefault(t['track_id'],dict(track_id=t['track_id'],first_timestamp_us=int(row['timestamp_us']),
                                                                last_timestamp_us=int(row['timestamp_us']),visible_frames=0,sources=Counter()))
                    stat['last_timestamp_us']=int(row['timestamp_us']);stat['visible_frames']+=1;stat['sources'][t['source']]+=1
                rendered=render(directory,row,path,image,ds,event,config,font,geometry)
                rendered.save(out/'overlays'/(path.stem+'.jpg'),quality=86)
                encoder.stdin.write(rendered.tobytes())
        finally:
            encoder.stdin.close(); status=encoder.wait()
        if status:
            raise RuntimeError((out/'encoding.log').read_text())
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                     '-show_entries','stream=nb_read_frames,width,height,codec_name','-of','json',str(out/'comparison_slow.mp4')],text=True))['streams'][0]
    assert int(probe['nb_read_frames'])==len(records)
    dump(out/'video_metadata.json',probe)
    dump(out/'frames.json',output_records)
    dump(out/'track_lifetimes.json',list(id_stats.values()))
    dump(out/'lifecycle_events.json',tracker.events)
    table(out/'tracks.csv',flat_tracks,['camera','group','image','frame_index','timestamp_us','track_id','source',
                                       'segment_id','segment_length','segment_frame_offset',
                                       'detection_confirmed_this_frame','last_detection_confidence','frames_since_detection',
                                       'unobserved_frames','x1','y1','x2','y2'])
    table(out/'frame_summary.csv',[dict(camera=camera,group=r['group'],image=r['image'],timestamp_us=r['timestamp_us'],
                                        baseline_boxes_025=r['baseline_boxes_025'],high_boxes_before_edge=r['high_threshold_boxes_before_edge'],
                                        high_boxes_after_filters=len(r['high_confidence_after_filters']),stable_tracks=len(r['tracks']),
                                        detector_tracks=sum(t['source'].startswith('detection') for t in r['tracks']),
                                        flow_tracks=sum(t['source']=='optical_flow' for t in r['tracks']),
                                        predicted_tracks=sum(t['source']=='prediction_hold' for t in r['tracks']),
                                        tentative_tracks=r['tentative_tracks'],edge_rejected=sum('edge' in d['reason'] for d in r['rejected_detections']))
                                    for r in output_records])
    selected=np.linspace(0,len(paths)-1,min(6,len(paths)),dtype=int).tolist()
    contact=Image.new('RGB',(1536,456*len(selected)),'#151515')
    for i,index in enumerate(selected):
        contact.paste(Image.open(out/'overlays'/(paths[index].stem+'.jpg')),(0,i*456))
    contact.save(out/'contact_sheet.jpg',quality=90)
    summary=dict(camera=camera,group=directory.name,**dict(totals),created_tracks=tracker.next_id-1,
                 rotation_filter=selection,eligible_segments=len(segments),
                 confirmed_track_ids=len(id_stats),confirmed_tracks_ending_on_ball_edge=sum(r['confirmed'] and r['reason']=='ball_edge_exit' for r in tracker.terminated),
                 lifecycle_termination_reasons=dict(Counter(r['reason'] for r in tracker.terminated)),
                 video_verified_frames=int(probe['nb_read_frames']),unlabeled_accuracy_metrics=None,
                 note='Counts describe postprocessing, not measured precision/recall or identity accuracy.')
    dump(out/'summary.json',summary)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=checkpoint_hash:
        raise ValueError('Checkpoint changed during inference')
    print('DONE',camera,directory.name,json.dumps(dict(totals)),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--camera',choices=['F3E','F59','all'],required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--device',default='0')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/inference_tracking.json')
    parser.add_argument('--replay-cache',action='store_true')
    parser.add_argument('--birth-conf',type=float)
    parser.add_argument('--continuation-conf',type=float)
    parser.add_argument('--edge-margin',type=float)
    parser.add_argument('--confirm-hits',type=int)
    parser.add_argument('--max-missing',type=int)
    parser.add_argument('--min-track-frames',type=int)
    parser.add_argument('--consecutive-max-gap-us',type=int)
    parser.add_argument('--ball-radius',type=float)
    parser.add_argument('--center-x',type=float,default=128.)
    parser.add_argument('--center-y',type=float,default=128.)
    parser.add_argument('--input-fps',type=float,default=2000.)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text()) if args.config.exists() else {}
    for argument,key in [('birth_conf','birth_confidence'),('continuation_conf','continuation_confidence'),
                         ('edge_margin','edge_margin_pixels'),('confirm_hits','confirmation_hits'),('max_missing','max_unobserved_frames'),
                         ('min_track_frames','minimum_consecutive_frames'),('consecutive_max_gap_us','consecutive_max_gap_us')]:
        value=getattr(args,argument)
        if value is not None:
            settings[key]=value
    config=TrackerConfig(**settings)
    out=args.output or ROOT/'experiments/tracking_v1'/('apply_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'tracker_config.json',asdict(config))
    summaries=[]
    cameras=['F3E','F59'] if args.camera=='all' else [args.camera]
    if args.source is not None and len(cameras)!=1:
        raise ValueError('A custom source requires one camera')
    for camera in cameras:
        model=None
        if not args.replay_cache:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(2)
            model=YOLO(str(EXP/camera/'train/weights/last.pt'))
            assert model.model.model[0].conv.in_channels==1
        sources=[args.source] if args.source else sorted((DATA/'unlabeled_inference'/camera).glob('serve*'))
        for source in sources:
            name=source.parent.name if source.name=='binary' else source.stem
            summaries.append(run_sequence(camera,source,out/camera/name,config,args,model=model))
    all_rotation_rows=[];all_segments=[]
    for path in sorted(out.glob('*/*/rotation_tracks.csv')):
        all_rotation_rows.extend(rows(path))
        all_segments.extend(json.loads((path.parent/'rotation_segments.json').read_text()))
    table(out/'rotation_tracks_all.csv',all_rotation_rows,ROTATION_FIELDS)
    dump(out/'rotation_segments_all.json',all_segments)
    dump(out/'summary.json',dict(status='completed',groups=summaries,total_frames=sum(g['frames'] for g in summaries),
                                config=asdict(config),no_model_retraining=True,
                                eligible_rotation_segments=len(all_segments),eligible_rotation_samples=len(all_rotation_rows),
                                unavailable_metrics=['false_positive_rate','recall','IDF1','identity_switches'],
                                reason='No independent human detection/identity ground truth; no claim of guaranteed tracking.'))
    print('ALL COMPLETED',out,flush=True)


if __name__=='__main__':
    main()
