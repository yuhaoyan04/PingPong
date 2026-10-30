"""Re-export saved tracks as >=3-frame observed segments, with full-frame review media."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import subprocess

from common import ROOT,DATA,dump,table
from track_segments import select_consecutive_segments,write_rotation_exports,rotation_rows,ROTATION_FIELDS
from apply_model import paint
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def render(directory,frame,before,after,min_frames=3):
    path=directory/'binary'/frame['image']
    ref=directory/'polarity_reference'/frame['image']
    im=Image.new('RGB',(1152,456),'#151515')
    base=Image.open(path).convert('RGB').resize((384,384),Image.Resampling.NEAREST)
    im.paste(base,(0,72));im.paste(base,(384,72))
    im.paste(Image.open(ref).convert('RGB').resize((384,384),Image.Resampling.NEAREST),(768,72))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    draw=ImageDraw.Draw(im)
    draw.text((5,2),f"{frame['group']} frame={frame['frame_index']} t={frame['timestamp_us']} us",fill='white',font=font)
    draw.text((5,25),'Offline filter | detector / verified optical flow only | prediction holds excluded',fill='#dddddd',font=font)
    titles=[f'Before: {len(before)} tracker boxes',f'Eligible >={min_frames}-frame segments: {len(after)}','Polarity reference + eligible segments']
    for i,title in enumerate(titles):draw.text((i*384+5,49),title,fill='white',font=font)
    paint(im,before,'cyan',font,(0,72),tracked=True)
    paint(im,after,'#00ff80',font,(384,72),tracked=True)
    paint(im,after,'#00ff80',font,(768,72),tracked=True)
    return im


def verify(frames,segments,max_gap_us,minimum):
    by_id=defaultdict(list)
    for ordinal,f in enumerate(frames):
        for t in f['tracks']:
            assert t['source']!='prediction_hold'
            assert t['rotation_eligible'] and t['segment_length']>=minimum
            by_id[t['segment_id']].append((ordinal,f['frame_index'],f['timestamp_us'],t))
    assert len(by_id)==len(segments)
    for segment in segments:
        points=by_id[segment['segment_id']]
        assert len(points)==segment['segment_length']>=minimum
        assert [p[3]['segment_frame_offset'] for p in points]==list(range(len(points)))
        for a,b in zip(points,points[1:]):
            assert b[0]==a[0]+1 and b[1]==a[1]+1 and 0<b[2]-a[2]<=max_gap_us


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=__import__('pathlib').Path,default=ROOT/'experiments/tracking_v1/comparison')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'experiments/tracking_min3_v1/filtered')
    parser.add_argument('--min-frames',type=int,default=3)
    parser.add_argument('--max-gap-us',type=int,default=750)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    review=args.output/'review';(review/'contact_sheets').mkdir(parents=True)
    totals=Counter();stats=[];combined=[];all_segments=[]
    files=sorted(args.input.glob('*/*/frames.json'))
    if not files:raise ValueError('Expected camera/group/frames.json under input')
    for source in files:
        camera=source.parent.parent.name;group=source.parent.name
        raw_source=source.parent/'raw_frames.json'
        raw_source=raw_source if raw_source.exists() else source
        raw=json.loads(raw_source.read_text())
        filtered,segments,rejected,summary=select_consecutive_segments(raw,args.min_frames,args.max_gap_us)
        verify(filtered,segments,args.max_gap_us,args.min_frames)
        out=args.output/camera/group;(out/'overlays').mkdir(parents=True)
        write_rotation_exports(out,filtered,segments,rejected,summary)
        dump(out/'frames.json',filtered)
        dump(out/'raw_frames.json',raw)
        table(out/'tracks.csv',rotation_rows(filtered),ROTATION_FIELDS)
        dump(out/'input_provenance.json',dict(input=str(raw_source.resolve()),input_sha256=hashlib.sha256(raw_source.read_bytes()).hexdigest(),
                                              tracker_recomputed=False,selection_offline=True))
        raw_directory=DATA/'unlabeled_inference'/camera/group
        with (out/'encoding.log').open('wb') as log:
            encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                       '-s','1152x456','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22',
                       '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(out/'comparison_slow.mp4')],stdin=subprocess.PIPE,stderr=log)
            try:
                for before,after in zip(raw,filtered):
                    image=render(raw_directory,after,before['tracks'],after['tracks'],args.min_frames)
                    image.save(out/'overlays'/after['image'].replace('.png','.jpg'),quality=86)
                    encoder.stdin.write(image.tobytes())
            finally:
                encoder.stdin.close();code=encoder.wait()
            if code:raise RuntimeError((out/'encoding.log').read_text())
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                         '-show_entries','stream=nb_read_frames,width,height,codec_name','-of','json',str(out/'comparison_slow.mp4')],text=True))['streams'][0]
        assert int(probe['nb_read_frames'])==len(raw)
        dump(out/'video_metadata.json',probe)
        indices=np.linspace(0,len(raw)-1,min(6,len(raw)),dtype=int).tolist()
        sheet=Image.new('RGB',(1152,456*len(indices)),'#151515')
        for i,index in enumerate(indices):
            sheet.paste(Image.open(out/'overlays'/raw[index]['image'].replace('.png','.jpg')),(0,i*456))
        sheet.save(out/'contact_sheet.jpg',quality=90)
        sheet.save(review/'contact_sheets'/(group+'.jpg'),quality=90)
        record=dict(camera=camera,group=group,frames=len(raw),**summary)
        dump(out/'summary.json',record);stats.append(record)
        totals.update({k:v for k,v in summary.items() if isinstance(v,int) and not isinstance(v,bool) and k not in ('minimum_consecutive_frames','max_consecutive_frame_gap_us')})
        totals['frames']+=len(raw)
        combined.extend(rotation_rows(filtered));all_segments.extend(segments)
        print('DONE',camera,group,'raw',summary['raw_track_samples'],'kept',summary['retained_samples'],'segments',summary['retained_segments'],flush=True)
    table(args.output/'rotation_tracks_all.csv',combined,ROTATION_FIELDS)
    dump(args.output/'rotation_segments_all.json',all_segments)
    table(review/'group_summary.csv',[{k:v for k,v in s.items() if isinstance(v,(str,int)) and not isinstance(v,bool)} for s in stats])
    dump(args.output/'summary.json',dict(status='completed',groups=stats,totals=dict(totals),minimum_consecutive_frames=args.min_frames,
                                        max_consecutive_frame_gap_us=args.max_gap_us,eligible_segments_verified=True,
                                        source='saved per-frame tracks, no model retraining or tracking recomputation',
                                        rotation_speed_computed=False))
    report=['三帧连续轨迹后处理','',
            f'已处理 {len(stats)} 组 / {totals["frames"]} 帧，原追踪结果 {totals["raw_track_samples"]} 个逐帧轨迹样本。',
            f'保留 {totals["retained_segments"]} 段、{totals["retained_samples"]} 个样本；每段至少连续 {args.min_frames} 帧。',
            f'剔除 {totals["discarded_short_segments"]} 段短于 {args.min_frames} 帧的片段（{totals["discarded_short_samples"]} 个观测样本）。',
            f'另有 {totals["prediction_only_samples_excluded"]} 个纯 prediction_hold 补帧不进入后续测量数据。',
            '', '保留规则',
            '同一相机、发球、track ID；在处理顺序和原 frame_index 上均相邻；时间戳严格递增，帧间隔不超过配置阈值。',
            f'当前最大帧间隔 {args.max_gap_us} us；当前数据实际采样间隔 500 us。',
            'detection_high、detection_low、经图像证据验证的 optical_flow 可计数；纯预测保持帧不计数，且将轨迹切段。',
            '同一 ID 总共出现三次但不连续，不保留；长轨迹后面偶发的一帧/两帧尾巴也单独剔除。',
            '满足条件后保留该连续片段的全部点，含最初两帧，使用原时间戳，不补造观测。',
            '这是离线完整片段筛选；原追踪器仍只使用当前与过去图像。实时应用须缓冲至少两帧后再确认前两帧。',
            '', '用于后续测转速的数据',
            'rotation_tracks_all.csv：全部合格逐帧坐标；必须按 segment_id 分组，不能只按 track_id 将不同段拼接。',
            'rotation_segments_all.json：每段帧数、时间范围、来源等信息。',
            '<相机>/<发球>/rotation_tracks.csv、rotation_segments.json：分组导出。',
            '坐标是检测/光流更新后的 Kalman 平滑估计；最少三点是数据筛选条件，并非已验证转速精度。',
            '', '人工审核',
            'review/contact_sheets/：各组六帧抽样对照。',
            '<相机>/<发球>/comparison_slow.mp4：全部帧 80 倍慢放，左侧原追踪、中间合格轨迹、右侧极性参考。',
            'frames.json 和 tracks.csv 仅含合格轨迹；raw_frames.json 只供诊断，不用于后续测转速。',
            'rejected_segments.json 记录被剔除的短段及原因。',
            '', '默认应用入口 scripts/predict_latest.sh 也已接入同样筛选，默认最少三帧。',
            '可追加 --min-track-frames 5 提高要求；不能设为低于 3 帧。单图不会导出可用于测转速的轨迹。',
            '配置位置：configs/inference_tracking.json。原始数据、模型及先前对照结果均未覆盖。']
    (args.output/'REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('ALL COMPLETED',dict(totals),flush=True)


if __name__=='__main__':main()
