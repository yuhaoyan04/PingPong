"""Audit exported clip times without claiming unmeasured hardware synchrony."""
from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .marker_spin import sha256
from .motion import write_csv
from .radius_review import font, CachedDraw


def manifest_map(rows):
    result={}
    previous=None
    for row in rows:
        t=int(row['timestamp_us'])
        if t in result or (previous is not None and t<=previous):
            raise ValueError('duplicate or non-increasing manifest timestamp')
        result[t]=row
        previous=t
    return result


def exact_pairs(left, right):
    """Use absolute timestamp intersection, never row-number alignment."""
    a,b=manifest_map(left),manifest_map(right)
    result=[]
    for t in sorted(a.keys() & b.keys()):
        result.append({'timestamp_us':t,'F3E_timestamp_us':int(a[t]['timestamp_us']),
                       'F59_timestamp_us':int(b[t]['timestamp_us']),'timestamp_delta_us':0,
                       'F3E_frame_index':int(a[t]['frame_index']),'F59_frame_index':int(b[t]['frame_index']),
                       'F3E_image':a[t]['image'],'F59_image':b[t]['image']})
    return result


def ffmpeg_writer(path, width, height, fps):
    log=path.with_suffix('.log').open('w')
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=log)
    return process,log


def finish(process, log):
    process.stdin.close()
    status=process.wait()
    log.close()
    if status:raise RuntimeError('ffmpeg failed during sync review')


def build_sync_review(dataset, output):
    dataset,output=Path(dataset),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name in ['paired','full_view','posters']:(output/name).mkdir(exist_ok=True)
    summaries=[];pairs_all=[];availability_all=[];reference_pairs=[]
    groups=sorted(p.name.rsplit('_',1)[0] for p in (dataset/'unlabeled_inference/F3E').iterdir() if p.is_dir())
    for group in groups:
        folders={};maps={};meta={};hashes={}
        for camera in ['F3E','F59']:
            folder=dataset/'unlabeled_inference'/camera/f'{group}_{camera}'
            folders[camera]=folder
            meta[camera]=json.loads((folder/'metadata.json').read_text())
            with (folder/'manifest.csv').open(encoding='utf-8-sig') as f:maps[camera]=manifest_map(list(csv.DictReader(f)))
            hashes[camera]={name:sha256(folder/name) for name in ['manifest.csv','metadata.json']}
        if any(meta['F3E'][k]!=meta['F59'][k] for k in ['video_first_us','video_step_us','reference_video']):
            raise ValueError(f'{group}: camera reference time mappings differ')
        pairs=exact_pairs(list(maps['F3E'].values()),list(maps['F59'].values()))
        union=sorted(maps['F3E'].keys()|maps['F59'].keys())
        for p in pairs:pairs_all.append({'serve':group,**p})
        for t in union:
            availability_all.append({'serve':group,'timestamp_us':t,'F3E_exported_crop_exists':t in maps['F3E'],
                                     'F59_exported_crop_exists':t in maps['F59'],
                                     'both_exported_crops_exist':t in maps['F3E'] and t in maps['F59']})
        if not pairs:raise ValueError(f'{group}: no common crop timestamps')
        process,log=ffmpeg_writer(output/'paired'/f'{group}_exact_time.mp4',960,560,25)
        for i,pair in enumerate(pairs):
            t=pair['timestamp_us'];canvas=Image.new('RGB',(960,560),'#121e2a');draw=CachedDraw(canvas)
            draw.text((18,12),f'{group}  双侧裁剪的共同时间段',font=font(25),fill='#e4edf5')
            draw.text((18,52),f'相同标注时间 {t} μs   |   标签差 0 μs',font=font(23),fill='#89e2b5')
            for j,camera in enumerate(['F3E','F59']):
                x=60+j*480;row=maps[camera][t]
                image=Image.open(folders[camera]/row['image']).convert('RGB').resize((320,320))
                canvas.paste(image,(x,145))
                draw.text((x,95),f"{camera} 源帧 {row['frame_index']}",font=font(20),fill='#f0ce89')
                draw.text((x,121),f"t={row['timestamp_us']} μs",font=font(16),fill='#ccd8e1')
                draw.text((x,477),row['track_support'],font=font(16),fill='#b4c8d9')
            draw.text((18,510),'按绝对时间戳交集配对，不将两侧第 0 帧强行对齐。',font=font(20),fill='#cddce7')
            draw.text((18,538),'这是导出层时间一致性；两台相机原始时钟同步等待发球 RAW 核验。',font=font(15),fill='#f0ce89')
            if i==len(pairs)//2:canvas.save(output/'posters'/f'{group}_paired.png')
            process.stdin.write(np.asarray(canvas).tobytes())
        finish(process,log)
        reference=dataset/meta['F3E']['reference_video']
        cap=cv2.VideoCapture(str(reference))
        if not cap.isOpened():raise RuntimeError(reference)
        process,log=ffmpeg_writer(output/'full_view'/f'{group}_full_reference.mp4',1280,460,30)
        index=0;first=meta['F3E']['video_first_us'];step=meta['F3E']['video_step_us']
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame.shape[:2]!=(720,2560):raise ValueError('expected original dual-camera full-view frame')
            t=first+index*step
            reference_pairs.append({'serve':group,'reference_frame_index':index,
                                    'F3E_reference_frame_index':index,'F59_reference_frame_index':index,
                                    'declared_F3E_timestamp_us':t,'declared_F59_timestamp_us':t,
                                    'physical_camera_time_verified':False})
            canvas=Image.new('RGB',(1280,460),'#121e2a');draw=CachedDraw(canvas)
            draw.text((12,8),f'{group} 原双路参考帧 {index} | 标注源时间 {t} μs | 原采样 {step} μs',font=font(20),fill='#e0ecf5')
            draw.text((12,39),'直接保留原视频同一帧中的两个面板，未对任何一侧单独移帧。',font=font(17),fill='#89e2b5')
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((1280,360))
            idraw=ImageDraw.Draw(image)
            for j,camera in enumerate(['F3E','F59']):
                row=maps[camera].get(t)
                if row:
                    x=float(row['center_x_px'])*.5+j*640;y=float(row['center_y_px'])*.5;r=float(row['radius_px'])*.5
                    idraw.ellipse((x-r,y-r,x+r,y+r),outline='#f7c666',width=2)
                text=f'{camera}：有该时刻导出 ROI' if row else f'{camera}：无该时刻导出 ROI'
                draw.text((12+j*640,426),text,font=font(17),fill='#89e2b5' if row else '#f0ce89')
            canvas.paste(image,(0,66))
            if index==int(round((pairs[len(pairs)//2]['timestamp_us']-first)/step)):
                canvas.save(output/'posters'/f'{group}_full.png')
            process.stdin.write(np.asarray(canvas).tobytes());index+=1
        cap.release();finish(process,log)
        summary={'serve':group,'F3E_crop_start_us':min(maps['F3E']),'F3E_crop_end_us':max(maps['F3E']),
                 'F59_crop_start_us':min(maps['F59']),'F59_crop_end_us':max(maps['F59']),
                 'F3E_crop_frames':len(maps['F3E']),'F59_crop_frames':len(maps['F59']),
                 'union_frames':len(union),'shared_frames':len(pairs),'single_crop_frames':len(union)-len(pairs),
                 'shared_first_us':pairs[0]['timestamp_us'],'shared_last_us':pairs[-1]['timestamp_us'],
                 'first_shared_F3E_frame':pairs[0]['F3E_frame_index'],'first_shared_F59_frame':pairs[0]['F59_frame_index'],
                 'reference_frames':index,'reference_first_us':first,'reference_step_us':step,
                 'shared_times_contiguous_500us':bool(np.all(np.diff([p['timestamp_us'] for p in pairs])==500)),
                 'source_hashes':hashes,'reference_video_sha256':sha256(reference)}
        summaries.append(summary)
        print(group,'union',len(union),'common',len(pairs),'reference',index,flush=True)
    write_csv(output/'exact_frame_pairs.csv',pairs_all)
    write_csv(output/'crop_availability.csv',availability_all)
    write_csv(output/'clip_summary.csv',summaries)
    write_csv(output/'reference_frame_pairs.csv',reference_pairs)
    report={'status':'exported_timestamp_pairing_checked_hardware_sync_pending_raw', 'serves':summaries,
            'union_frames':len(availability_all),'shared_frames':len(pairs_all),
            'single_crop_frames':len(availability_all)-len(pairs_all),'all_exact_pair_timestamp_deltas_us':0,
            'physical_camera_synchronization_verified':False,'ball_raw_upload_pending':True,
            'reference_video_times_source':'metadata.video_first_us + frame_index * video_step_us',
            'reference_labels_independently_verified':False,
            'interpretation':'Missing exported crop is not evidence that the original camera did not see a ball. Original references can also have one view without a visible ball; physical time offset versus field-of-view geometry requires ball RAW verification.'}
    if (output/'reference_timecode_report.json').is_file():
        report['reference_label_audit']=json.loads((output/'reference_timecode_report.json').read_text())
    (output/'sync_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双视角时间与裁剪范围审核</title><style>body{font:16px system-ui;max-width:1300px;margin:25px auto;padding:0 18px;background:#f4f6f9;color:#192332;line-height:1.6}.note{background:#fff0cd;padding:15px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}video{display:block;max-width:100%;margin:12px auto}section{background:white;padding:18px;margin:20px 0}a{color:#185b99}</style><h1>两视角时间戳与裁剪导出范围审核</h1>
<p class="note">这里验证的是导出层配对，不是两台相机物理时钟的独立证明。对应发球 RAW 正在等待上传，硬件时间域、延迟和原始事件时间完整性仍待核验。</p>'''
    page+=f'<p>此前视频使用两侧导出裁剪的时间并集：{report["union_frames"]} 个时刻中仅 {report["shared_frames"]} 个有双侧裁剪，其余 {report["single_crop_frames"]} 个缺少一侧导出 ROI。黑面板应理解为“没有此时刻的导出裁剪”，不能等同于“相机当时没拍到球”。</p>'
    page+='<p>每次发球提供两段视频：原全视野参考视频保留原文件同一帧中的两个面板；裁剪审核只保留绝对时间戳严格相同的交集，不平移单侧画面来让球看起来同时出现。原全视野自身也有部分时刻只在一侧可见球；视场差异与物理时钟偏差尚未区分。</p><p><a href="exact_frame_pairs.csv">逐帧精确配对 CSV</a> · <a href="reference_frame_pairs.csv">原参考帧映射 CSV</a> · <a href="crop_availability.csv">全部裁剪有无记录</a> · <a href="sync_report.json">完整审计 JSON</a></p>'
    page+='<table><tr><th>发球</th><th>F3E / F59 裁剪数</th><th>共同 / 并集</th><th>第一对源帧 F3E / F59</th><th>共同起点 μs</th></tr>'
    for s in summaries:
        page+=f'<tr><td>{s["serve"]}</td><td>{s["F3E_crop_frames"]} / {s["F59_crop_frames"]}</td><td>{s["shared_frames"]} / {s["union_frames"]}</td><td>{s["first_shared_F3E_frame"]} / {s["first_shared_F59_frame"]}</td><td>{s["shared_first_us"]}</td></tr>'
    page+='</table>'
    for s in summaries:
        name=s['serve']
        page+=f'<section><h2>{html.escape(name)}</h2><p>原全视野：5 ms 取样、30 FPS 播放。相机标签依据本批历史导出映射：左 F3E、右 F59；不套用之后的主从角色配置。</p><video controls preload="metadata" poster="posters/{name}_full.png" src="full_view/{name}_full_reference.mp4"></video><p>共同裁剪：0.5 ms 取样、25 FPS 播放；两侧各自显示绝对时间和源帧号。两段视频采样率和起点不同，不能用播放进度条直接互相对齐。</p><video controls preload="metadata" poster="posters/{name}_paired.png" src="paired/{name}_exact_time.mp4"></video></section>'
    page+='</html>'
    (output/'review.html').write_text(page,encoding='utf-8')
    return report
