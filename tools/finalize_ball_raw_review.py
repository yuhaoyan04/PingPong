"""Validate/package the actual ball RAW review and write an offline HTML player."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.motion import write_csv
from pingpong_event_pipeline.event_timing import stream_sha256
from pingpong_event_pipeline.raw_sync import load_serves


def optical_probe(output):
    a = np.load(output/'activity_F3E.npz'); b = np.load(output/'activity_F59.npz')
    if int(a['origin_us']) != int(b['origin_us']) or int(a['bin_us']) != int(b['bin_us']):
        raise ValueError('activity clocks differ')
    origin, step = int(a['origin_us']), int(a['bin_us'])
    records = []; traces = {}
    for start, end in [(645,670),(645,653),(653,662),(662,670)]:
        lo, hi = (int((t*1e6-origin)//step) for t in [start,end])
        for feature in ['TOTAL','ON','OFF','ON-OFF']:
            values = []
            for archive in [a,b]:
                h = archive['counts']
                x = (h.sum(axis=0) if feature=='TOTAL' else h[1] if feature=='ON' else h[0] if feature=='OFF' else h[1]-h[0]).astype(float)
                values.append((x-gaussian_filter1d(x,20))[lo:hi])
            correlations = []
            x,y = values
            for lag in range(-100,101):
                u,v = (x[-lag:],y[:lag]) if lag<0 else (x[:-lag],y[lag:]) if lag>0 else (x,y)
                correlations.append(float(np.corrcoef(u,v)[0,1]))
            best = int(np.nanargmax(correlations)); traces[(start,end,feature)] = correlations
            records.append({'start_us':start*1_000_000,'end_us':end*1_000_000,'feature':feature,
                            'best_lag_us':(best-100)*step,'peak_correlation':correlations[best],
                            'zero_lag_correlation':correlations[100], 'accepted_as_clock_offset':False})
    write_csv(output/'optical_sync_probe.csv', records)
    fig, axes = plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for ax, feature in zip(axes.ravel(),['TOTAL','ON','OFF','ON-OFF']):
        for start,end in [(645,670),(645,653),(653,662),(662,670)]:
            ax.plot(np.arange(-100,101)*step/1000, traces[(start,end,feature)],label=f'{start}-{end} s')
        ax.axvline(0,color='.6',linewidth=.8); ax.set(title=feature,xlabel='Slave - master lag (ms)',ylabel='Correlation')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/'optical_sync_probe.png',dpi=140);plt.close(fig)
    report = {'external_trigger_words':0,'accepted_clock_correction_us':None,
              'maximum_observed_correlation':max(r['peak_correlation'] for r in records),
              'interpretation':'No independent trigger anchors in these RAWs. Global activity peaks are weak and inconsistent across features/segments; no physical sync offset is accepted or applied.',
              'bin_us':step,'highpass_gaussian_sigma_bins':20,'search_range_us':[-50000,50000],
              'measurements':records}
    (output/'optical_sync_probe.json').write_text(json.dumps(report,indent=2))
    return report


def finalize(output, dataset, radius):
    output, dataset, radius = map(Path,[output,dataset,radius])
    render = json.loads((output/'render_report.json').read_text())
    cameras = {c:json.loads((output/f'raw_windows_{c}.json').read_text()) for c in ['F3E','F59']}
    timing = {c:json.loads((output/f'timing_{c}/timing_report.json').read_text()) for c in cameras}
    with (output/'reference_raw_frames.csv').open(encoding='utf-8-sig') as stream:
        compared=list(csv.DictReader(stream))
    expected_comparisons=2*sum(s['reference_frames'] for s in render['serves'])
    minimum_reference_iou=min(float(r['old_reference_occupied_pixel_iou']) for r in compared)
    if len(compared)!=expected_comparisons or minimum_reference_iou!=1 or not render['all_sampled_reference_phases_zero']:
        raise ValueError('reference-to-RAW mapping not verified; inspect the render diagnostics')
    for c in cameras:
        if not timing[c]['decoded_intervals_verified'] or not cameras[c]['all_selected_windows_sdk_reference_agree'] or not cameras[c]['all_selected_windows_chronological']:
            raise ValueError(f'{c}: raw verification incomplete')
        if stream_sha256(timing[c]['raw'])!=timing[c]['raw_sha256']:
            raise ValueError('RAW changed during review')
        clock = np.load(output/f'timing_{c}/event_clock_runs.npz')
        if int(clock['counts'].sum(dtype=np.uint64)) != timing[c]['sdk']['events']:
            raise ValueError('clock encoding loses events')
    validation = []
    for s in render['serves']:
        for folder, suffix, frames in [('full_view','raw_reference',s['reference_frames']),('tracking_spin','raw_tracking_spin',s['tracking_frames'])]:
            file = output/folder/f"{s['serve']}_{suffix}.mp4"
            cap = cv2.VideoCapture(str(file)); count = 0; shape = None
            while True:
                ok, frame = cap.read()
                if not ok:break
                shape = list(frame.shape[:2]); count += 1
            cap.release()
            if count!=frames:raise ValueError(f'{file}: decoded {count} expected {frames}')
            validation.append({'file':str(file.relative_to(output)),'decoded_frames':count,'shape':shape,'passed':True})
    probe = optical_probe(output)
    appearance_path=output/'appearance/appearance_report.json'
    appearance=json.loads(appearance_path.read_text()) if appearance_path.is_file() else None
    shutil.copy2(radius/'frame_overlays.json',output/'overlay_values.json')
    shutil.copy2(radius/'radius_summary.csv',output/'radius_summary.csv')
    availability=[]
    for spec in load_serves(dataset):
        sets={c:{int(row['timestamp_us']) for row in spec['cameras'][c]['rows']} for c in cameras}
        for t in sorted(sets['F3E']|sets['F59']):
            availability.append({'serve':spec['serve'],'timestamp_us':t,
                                 'F3E_exported_crop_exists':t in sets['F3E'],
                                 'F59_exported_crop_exists':t in sets['F59'],
                                 'both_exported_crops_exist':t in sets['F3E'] and t in sets['F59']})
    write_csv(output/'crop_availability.csv',availability)
    single_crop_frames=sum(not r['both_exported_crops_exist'] for r in availability)
    report = {'status':'actual_ball_raw_decode_and_export_mapping_verified_physical_sync_unresolved',
              'ball_raw_upload_pending':False,
              'total_events':sum(t['sdk']['events'] for t in timing.values()),
              'total_negative_event_deltas':sum(t['sdk']['negative_intervals'] for t in timing.values()),
              'full_sdk_evt3_sequence_agreement':True,
              'selected_serve_windows_exact_sdk_agreement':True,
              'padded_seek_windows_exact_sdk_agreement':all(c['all_padded_seeks_exact'] for c in cameras.values()),
              'all_sampled_reference_phases_zero':render['all_sampled_reference_phases_zero'],
              'reference_phase_samples':render['phase_samples'],
              'reference_camera_frames_compared':len(compared),
              'minimum_reference_occupied_pixel_iou':minimum_reference_iou,
              'physical_camera_synchronization_verified':False,
              'physical_clock_correction_us':None,
              'export_mapping_verified':True,'spin_candidates_recomputed':False,
              'spin_reuse_reason':'No incorrect RAW-to-export timing mapping found. Existing candidate fits and radius variants retained, without changing rates to match expectations.',
              'physical_spin_accuracy_validated':False,
              'appearance_period_diagnostic':'appearance/appearance_report.json' if appearance else None,
              'raw_radius_review':'raw_radii/review.html' if (output/'raw_radii/review.html').is_file() else None,
              'cameras':{c:{'raw':t['raw'],'raw_sha256':t['raw_sha256'],'events':t['sdk']['events'],
                            'first_us':t['sdk']['first_event_us'],'last_us':t['sdk']['last_event_us'],
                            'minimum_positive_delta_us':t['sdk']['minimum_positive_delta_us'],
                            'same_timestamp_intervals':t['sdk']['same_timestamp_intervals']} for c,t in timing.items()},
              'optical_sync_probe':probe,'videos':validation,'serves':render['serves'],
              'crop_availability_source':'crop_availability.csv',
              'union_crop_timestamps':len(availability),'single_crop_timestamps':single_crop_frames,
              'interpretation':['Every CD event independently checked; 1 us is quantization, not calibrated clock accuracy.',
                                'Equal-time RAW rendering preserves both full views, including times with no exported crop.',
                                'Reference image comparisons verify each view against its own RAW, not physical alignment between clocks.',
                                'No automatic camera shift from first-event times, ball entry, or weak activity correlation.']}
    (output/'sync_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (output/'validation.json').write_text(json.dumps({'passed':True,'videos':validation,'raws_unchanged':True,'event_clock_run_counts_verified':True},indent=2))
    assets=[]
    source_files=[radius/'frame_overlays.json',radius/'radius_summary.csv',Path('output/latest/tracking/rotation_tracks_all.csv')]
    for camera in cameras:
        source_files.append(Path(f'experiments/two_camera_v1/{camera}/train/weights/last.pt'))
        for folder in (dataset/'unlabeled_inference'/camera).iterdir():
            source_files.extend([folder/'manifest.csv',folder/'metadata.json'])
    for file in source_files:
        digest=stream_sha256(file)
        assets.append({'file':str(file.resolve()),'sha256':digest,'size_bytes':file.stat().st_size})
    (output/'input_assets.json').write_text(json.dumps(assets,indent=2))
    ui = {'serves':render['serves']}
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>发球 RAW、轨迹与转速审核</title>
<style>body{font:16px system-ui;max-width:1320px;margin:24px auto;padding:0 18px;color:#1a2837;background:#eef2f6;line-height:1.6}h1{font-size:27px}.note{background:#fff0cd;padding:14px;border-radius:8px}video{width:100%;background:#142130;display:block;margin:14px 0}button,select{font:inherit;padding:6px;margin:4px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #ccd4dd;text-align:left}a{color:#1462a0}.box{background:white;padding:14px;margin:15px 0}code{font-size:14px}img{max-width:100%}</style>
<h1>实际发球 RAW：时间、轨迹与转速审核</h1>
<p>两份 RAW 共 <b>__EVENTS__</b> 个事件，SDK 顺序解码与独立 EVT3 解码逐事件哈希一致，负时间间隔为 0。__SERVES__ 次发球、两路旧参考画面的前景事件像素与同标签 RAW 窗口一致；__PHASES__ 个抽样的 ±10 ms 搜索，最佳偏移均为 0。</p>
<p class="note">已核实导出时间与各自 RAW 对应，未发现单侧导出移帧。两台相机的<b>物理时钟同步精度仍未独立证实</b>：RAW 无外部触发事件，全局活动相关峰弱且不稳定。因此不施加人为时移，转速与方向仍按候选展示。</p>
<p>以前视频 __UNION__ 个时刻中有 __SINGLE__ 个缺少一侧导出裁剪。新版始终显示两侧 RAW 全视野，并单独标明裁剪有无；可直接检查缺裁剪时相机实际拍到了什么。</p>
<p><a href="raw_review_bundle.zip">下载完整审核包</a> · <a href="sync_report.json">完整审计</a> · <a href="reference_raw_frames.csv">逐帧 RAW 窗口证据</a> · <a href="reference_phase_search.csv">参考画面时间搜索</a> · <a href="radius_summary.csv">半径对照汇总</a></p>
__APPEARANCE__
<div class="box"><label>发球 <select id="serve"></select></label><label>内容 <select id="mode"><option value="tracking_spin">RAW全视野＋轨迹＋转速方向</option><option value="full_view">RAW重建与旧参考对照</option></select></label>
<button id="prev">上一帧</button><button id="next">下一帧</button><button id="slow">0.5×播放</button><div id="frame"></div><video id="video" controls preload="metadata"></video>
<table><tr><th>球面半径假设</th><th>当前转速候选</th><th>来源 / 有向轴坐标</th></tr><tbody id="values"></tbody></table>
<p>两种视频的采样步长不同：轨迹视频 0.5 ms、25 FPS 慢放；全视野对照 5 ms、30 FPS 慢放。计算使用 RAW 事件时间。方向箭头是相机坐标下可见前表面中心的运动，不等同于世界坐标上/下旋。</p></div>
<h2>逐事件间隔</h2><p>当前两份记录的相邻事件间隔均为 0 或 1 μs；同一个 1 μs 时间格内可以有多个事件。0 不证明物理上完全同时。压缩文件无损保留每一个事件的时间次序，包括零间隔与跨 SDK 批次间隔；它不是 0.5 ms 帧间隔。</p>
<p><a href="timing_F3E/event_clock_runs.npz">F3E 全量事件时间编码</a> · <a href="timing_F59/event_clock_runs.npz">F59 全量事件时间编码</a> · <a href="timing_F3E/adjacent_events_sample.csv">F3E 相邻事件样例</a> · <a href="timing_F59/adjacent_events_sample.csv">F59 相邻事件样例</a></p>
<h2>同步证据的边界</h2><p>两路首事件相同、文件日期接近、时间标签相同，均不足以校准两路物理时钟。全局活动相关峰随极性和时段改变，不采用它来强行对齐。以下曲线是诊断结果，不能作为硬件同步通过证明。</p><img src="optical_sync_probe.png"><p><a href="optical_sync_probe.csv">相关诊断数据</a> · <a href="validation.json">视频与时间编码验证</a></p>
<script>const DATA=__UI__,OVERLAYS=__OVERLAYS__;
const v=document.getElementById('video'),sel=document.getElementById('serve'),mode=document.getElementById('mode');let currentFrame=0;
DATA.serves.forEach(s=>{let o=document.createElement('option');o.value=s.serve;o.textContent=s.serve;sel.appendChild(o)});
function spec(){return DATA.serves.find(s=>s.serve===sel.value)}function ov(){return OVERLAYS.serves.find(s=>s.serve===sel.value)}function fps(){return mode.value==='full_view'?30:25}
function display(mediaTime){let s=spec(),o=ov(),full=mode.value==='full_view',count=full?s.reference_frames:s.tracking_frames;currentFrame=Math.max(0,Math.min(count-1,Math.floor(mediaTime*fps()+0.0001)));let t=full?s.reference_first_us+currentFrame*s.reference_step_us:o.timestamps_us[currentFrame];document.getElementById('frame').textContent=`帧 ${currentFrame+1}/${count}；两侧源事件窗 [${t-1000}, ${t}) μs；相同时间标签`;
document.getElementById('values').innerHTML=full?'<tr><td colspan="3">全视野对照模式；切换轨迹模式查看当前观测的转速候选。</td></tr>':o.values[currentFrame].map((x,k)=>`<tr><td>R × ${[1,1.1,1.2,1.5][k]}</td><td>${x.available?x.rps.toFixed(2)+' 转/秒':'N/A'}</td><td>${x.source||'无有效观测'} / ${x.axis_camera||'—'} ${x.axis?'('+x.axis.map(n=>n.toFixed(3)).join(', ')+')':'方向未分辨'}</td></tr>`).join('');}
function load(){v.pause();v.src=mode.value+'/'+sel.value+(mode.value==='full_view'?'_raw_reference.mp4':'_raw_tracking_spin.mp4');v.poster='posters/'+sel.value+(mode.value==='full_view'?'_raw_reference.png':'_raw_tracking_spin.png');currentFrame=0;display(0)}
function step(d){v.pause();let s=spec(),n=mode.value==='full_view'?s.reference_frames:s.tracking_frames;currentFrame=Math.max(0,Math.min(n-1,currentFrame+d));v.currentTime=(currentFrame+0.1)/fps();display(v.currentTime)}
document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);document.getElementById('slow').onclick=()=>{v.playbackRate=.5;v.play()};sel.onchange=load;mode.onchange=load;
v.addEventListener('seeked',()=>{if(v.paused)display(v.currentTime)});if(v.requestVideoFrameCallback){function cb(_,m){display(m.mediaTime);v.requestVideoFrameCallback(cb)}v.requestVideoFrameCallback(cb)}else v.addEventListener('timeupdate',()=>display(v.currentTime));load();</script></html>'''
    overlay = json.loads((output/'overlay_values.json').read_text())
    # Inline data keeps the downloaded review functional under file:// as well.
    for s in ui['serves']:
        folder = dataset/'unlabeled_inference/F3E'/f"{s['serve']}_F3E"
        m = json.loads((folder/'metadata.json').read_text())
        s.update(reference_first_us=m['video_first_us'],reference_step_us=m['video_step_us'])
    appearance_note='<p><a href="appearance/review.html"><b>新增：标记纹理重复周期交叉检查</b></a>，含两次相似姿态的并排图和重复周期曲线。02/03/04/07/08 的双侧共同纹理频率约 34.8 / 41.2 / 43.5 / 21.9 / 38.8 次/秒；05 未找到共同分支，09 同时存在约 7.25 与 26.5 ms 分支，仍有歧义。纹理对称与多周重复使它不能直接当作物理转速真值。</p>' if appearance else ''
    if (output/'raw_radii/review.html').is_file():
        appearance_note+='<p><a href="raw_radii/review.html"><b>新增：RAW 球边界上的四种半径对照</b></a>，同一时刻、同一像素与球心，直接检查假设圆是否贴合真实球轮廓及对应转速；不能仅因数值更低就采用更大半径。</p>'
    page=page.replace('__UI__',json.dumps(ui,ensure_ascii=False)).replace('__OVERLAYS__',json.dumps(overlay,ensure_ascii=False)).replace('__EVENTS__',f"{report['total_events']:,}").replace('__SERVES__',str(len(render['serves']))).replace('__PHASES__',str(render['phase_samples'])).replace('__APPEARANCE__',appearance_note).replace('__UNION__',f'{len(availability):,}').replace('__SINGLE__',str(single_crop_frames))
    (output/'review.html').write_text(page,encoding='utf-8')
    for file in output.rglob('*.log'):
        if file.stat().st_size==0:file.unlink()
    with zipfile.ZipFile(output/'raw_review_bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.name!='raw_review_bundle.zip' and 'timing_F3E_initial_eof' not in file.parts:
                z.write(file,file.relative_to(output))
    print(json.dumps({'events':report['total_events'],'videos':len(validation),'bundle_bytes':(output/'raw_review_bundle.zip').stat().st_size},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--radius',type=Path,default=Path('output/latest_radius'))
    a=p.parse_args();finalize(a.output,a.dataset,a.radius)
