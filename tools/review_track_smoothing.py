"""Compare original and robust offline marker trajectories at fixed geometry/windows."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pingpong_event_pipeline.marker_spin import read_tracks,sha256
from pingpong_event_pipeline.trajectory_smoothing import csv_rows
from pingpong_event_pipeline.serve_spin import analyze_serves
from pingpong_event_pipeline.radius_review import baseline_source,current_fit,frame_value,font,CachedDraw,arrow
from pingpong_event_pipeline.clip_sync import ffmpeg_writer,finish
from pingpong_event_pipeline.raw_sync import load_serves
from pingpong_event_pipeline.motion import write_csv


def roughness(p):
    return float(np.sqrt(np.mean(np.diff(p,3,axis=0)**2))) if len(p)>3 else None


def reversals(p):
    d=np.diff(p,axis=0)
    if len(d)<3:return 0
    ref=(d[:-2]+d[2:])/2;mid=d[1:-1]
    norms=np.linalg.norm(ref,axis=1)*np.linalg.norm(mid,axis=1)
    return int(np.count_nonzero((norms>.2)&(np.sum(ref*mid,axis=1)<-.3*norms)))


def build(tracks,dataset,config,smoothing,baseline,output,videos=True):
    tracks,dataset,output=map(Path,[tracks,dataset,output]);output.mkdir(parents=True,exist_ok=False)
    for folder in ['plots','videos','posters']:(output/folder).mkdir()
    raw,sources,_=read_tracks(tracks,dataset)
    filtered,filtered_sources,_=read_tracks(tracks,dataset,smoothing_config=smoothing)
    write_csv(output/'smoothed_rotation_tracks.csv',csv_rows(filtered))
    diagnostics=[]
    for key,run in raw.items():
        p=np.array([r['point'] for r in run]);q=np.array([r['point'] for r in filtered[key]])
        status=filtered_sources[f'{key[0]}/{key[1]}']['trajectory_smoothing'][key[2]]['status']
        diagnostics.append({'camera':key[0],'group':key[1],'segment_id':key[2],'observations':len(run),'status':status,
                            'raw_third_difference_rms_px':roughness(p),'smoothed_third_difference_rms_px':roughness(q),
                            'raw_local_reverse_steps':reversals(p),'smoothed_local_reverse_steps':reversals(q),
                            'median_correction_px':float(np.median(np.linalg.norm(p-q,axis=1))),
                            'maximum_correction_px':float(np.max(np.linalg.norm(p-q,axis=1)))})
    write_csv(output/'segment_metrics.csv',diagnostics)
    no_filter={k:v for k,v in config.items() if k!='trajectory_smoothing'}
    before=json.loads(Path(baseline).read_text())
    if before['tracks_sha256']!=sha256(tracks) or before['config']!=no_filter:
        before=analyze_serves(tracks,dataset,no_filter)
    fixed={s['serve']:{k:s[k] for k in ['R_F59_to_F3E','orientation_training_serves']} for s in before['serves']}
    (output/'before_spin.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
    after=analyze_serves(output/'smoothed_rotation_tracks.csv',dataset,no_filter,fixed_relative_orientations=fixed)
    (output/'after_spin.json').write_text(json.dumps(after,ensure_ascii=False,indent=2,allow_nan=False))
    print('Spin comparison complete',flush=True)
    bm={s['serve']:s for s in before['serves']};am={s['serve']:s for s in after['serves']}
    paired_windows=[]
    for name,b in bm.items():
        a=am[name]
        for wb,wa in zip(b['windows'],a['windows']):
            assert wb['window_id']==wa['window_id']
            for source in ['JOINT','F3E','F59']:
                x=wb if source=='JOINT' else wb['mono'][source];y=wa if source=='JOINT' else wa['mono'][source]
                axis_delta=None
                if x.get('candidate_axis_camera') is not None and y.get('candidate_axis_camera') is not None:
                    axis_delta=float(np.degrees(np.arccos(np.clip(np.dot(x['candidate_axis_camera'],y['candidate_axis_camera']),-1,1))))
                paired_windows.append({'serve':name,'window_id':wb['window_id'],'source':source,
                                       'before_usable':x['numerically_usable'],'after_usable':y['numerically_usable'],
                                       'before_rps':x.get('candidate_rps'),'after_rps':y.get('candidate_rps'),
                                       'signed_axis_change_degrees':axis_delta,
                                       'before_inlier_fraction':x.get('balanced_inlier_fraction',x.get('inlier_fraction')),
                                       'after_inlier_fraction':y.get('balanced_inlier_fraction',y.get('inlier_fraction'))})
    write_csv(output/'matched_spin_windows.csv',paired_windows)
    by_time={};runs={}
    for mode,segments in [('before',raw),('after',filtered)]:
        by_time[mode]=defaultdict(list);runs[mode]=defaultdict(list)
        for (camera,group,segment),run in segments.items():
            for r in run:by_time[mode][(group,r['timestamp'])].append(r)
            runs[mode][segment]=run
    for camera,group in sorted({k[:2] for k in raw}):
        fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for key,run in raw.items():
            if key[:2]!=(camera,group):continue
            p=np.array([r['point'] for r in run]);q=np.array([r['point'] for r in filtered[key]])
            axes[0].plot(p[:,0],p[:,1],'.-',color='tab:orange',alpha=.4,linewidth=1)
            axes[0].plot(q[:,0],q[:,1],'-',color='teal',linewidth=1.3)
            t=(np.array([r['timestamp'] for r in run])-min(x['timestamp'] for k,v in raw.items() if k[:2]==(camera,group) for x in v))/1000
            axes[1].plot(t,np.linalg.norm(q-p,axis=1),'.-',markersize=2)
        axes[0].set(xlim=(0,256),ylim=(256,0),aspect='equal',title=group+' | orange original / teal smoothed',xlabel='crop x (px)',ylabel='crop y (px)')
        axes[1].set(title='Displacement, not ground-truth error',xlabel='Source time since first observation (ms)',ylabel='Smoothing correction (px)')
        for ax in axes:ax.grid(alpha=.2)
        fig.savefig(output/'plots'/f'{group}.png',dpi=140);plt.close(fig)
    overlays=[]
    for spec in load_serves(dataset):
        name=spec['serve'];maps={c:{int(r['timestamp_us']):r for r in spec['cameras'][c]['rows']} for c in ['F3E','F59']}
        times=sorted(maps['F3E'].keys()|maps['F59'].keys());values=[]
        for t in times:
            source=baseline_source(bm[name],t);fit=current_fit(bm[name],t,source) if source else None
            wid=fit[0]['window_id'] if fit else None
            values.append([frame_value(bm[name],t,source,wid),frame_value(am[name],t,source,wid)])
        def render(index):
            t=times[index];canvas=Image.new('RGB',(1280,1040),'#142130');draw=CachedDraw(canvas)
            draw.text((12,8),f'{name} | t={t} μs | 同一图像、时间、轨迹 ID、球半径与相机方向',font=font(22),fill='white')
            draw.text((12,44),'左：原 Kalman 轨迹；右：追加鲁棒局部三次平滑。灰线保留原路径，彩点是当前中心。',font=font(18),fill='#eacb87')
            for j,c in enumerate(['F3E','F59']):
                row=maps[c].get(t);y=128+j*410;group=f'{name}_{c}'
                for k,mode in enumerate(['before','after']):
                    x=k*640+18;value=values[index][k]
                    draw.text((x,y-33),f'{c} | '+('原轨迹' if k==0 else '鲁棒曲线平滑'),font=font(22),fill='#efb270' if k==0 else '#95e5c3')
                    if row:
                        pic=Image.open(Path(spec['cameras'][c]['folder'])/row['image']).convert('RGB').resize((320,320))
                        canvas.paste(pic,(x,y))
                        for r in by_time[mode].get((group,t),[]):
                            original=runs['before'][r['segment_id']]
                            prior=[p for p in original if 0<=t-p['timestamp']<=5000]
                            oldxy=[(x+p['point'][0]*1.25,y+p['point'][1]*1.25) for p in prior]
                            if len(oldxy)>1:draw.line(oldxy,fill='#808080',width=2)
                            trail=[p for p in runs[mode][r['segment_id']] if 0<=t-p['timestamp']<=5000]
                            xy=[(x+p['point'][0]*1.25,y+p['point'][1]*1.25) for p in trail]
                            col='#ffab56' if k==0 else '#69f5c8'
                            if len(xy)>1:draw.line(xy,fill=col,width=2)
                            px,py=x+r['point'][0]*1.25,y+r['point'][1]*1.25
                            draw.ellipse((px-4,py-4,px+4,py+4),fill=col)
                            draw.text((px+5,py-15),str(r['track_id']),font=font(14),fill=col)
                    else:
                        draw.rectangle((x,y,x+320,y+320),fill='#080e16');draw.text((x+30,y+140),'此时刻无导出裁剪',font=font(20),fill='#eacb87')
                    rate=f"{value['rps']:.2f} 转/秒" if value['available'] else 'N/A'
                    draw.text((x+350,y+22),rate,font=font(26),fill='#efb270' if k==0 else '#95e5c3')
                    draw.text((x+350,y+72),str(value['source'] or '无当前有效观测'),font=font(19),fill='#d1dfeb')
                    arrow(draw,(x+440,y+175),value.get('directions',{}).get(c),'#95e5c3')
                    draw.text((x+350,y+239),f'{c} 前表面运动方向',font=font(17),fill='#d1dfeb')
                    if mode=='after':
                        correction=max((float(r['smoothing_displacement_px']) for r in by_time[mode].get((group,t),[])),default=0)
                        draw.text((x,y+331),f'当前最大中心修正 {correction:.2f} px；不增加观测、不跨段平滑',font=font(17),fill='#d1dfeb')
            draw.text((12,958),'两列锁定同一个旋转窗口与数据来源；若平滑后该窗不再合格，显示 N/A，不换成其它窗口。',font=font(18),fill='#eacb87')
            draw.text((12,997),'500 μs 源采样，25 FPS 慢放；平滑使用未来帧，转速/方向仍未通过独立物理精度标定。',font=font(18),fill='#d1dfeb')
            return canvas
        representative=max(range(len(times)),key=lambda i:(sum(v['available'] for v in values[i]),-abs(i-len(times)/2)))
        render(representative).save(output/'posters'/f'{name}.png')
        if videos:
            process,log=ffmpeg_writer(output/'videos'/f'{name}_smoothing.mp4',1280,1040,25)
            for i in range(len(times)):process.stdin.write(np.asarray(render(i)).tobytes())
            finish(process,log)
        overlays.append({'serve':name,'frames':len(times),'timestamps_us':times,'values':values})
        print(name,'rendered',len(times),flush=True)
    (output/'frame_values.json').write_text(json.dumps({'serves':overlays},ensure_ascii=False))
    all_rows=[r for run in filtered.values() for r in run]
    summary={'method':'offline robust local cubic with leverage-corrected Huber IRLS and source/shape weights',
             'config':smoothing,'observations':len(all_rows),'segments':len(filtered),
             'segment_status':dict(Counter(r['status'] for r in diagnostics)),
             'smoothed_observations':sum(r['trajectory_smoothing_applied'] for r in all_rows),
             'median_correction_px':float(np.median([r['smoothing_displacement_px'] for r in all_rows])),
             'maximum_correction_px':max(r['smoothing_displacement_px'] for r in all_rows),
             'raw_local_reverse_steps':sum(r['raw_local_reverse_steps'] for r in diagnostics),
             'smoothed_local_reverse_steps':sum(r['smoothed_local_reverse_steps'] for r in diagnostics),
             'input_tracks_sha256':sha256(tracks),'effective_tracks_sha256':sha256(output/'smoothed_rotation_tracks.csv'),
             'identities_timestamps_observation_count_unchanged':True,'orientation_fixed_to_baseline':True,
             'physical_accuracy_validated':False,'source_diagnostics':filtered_sources}
    summary['videos_generated']=videos
    (output/'smoothing_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>标记中心轨迹平滑对照</title><style>body{font:16px system-ui;max-width:1300px;margin:25px auto;line-height:1.7;color:#192a3b}video,img{max-width:100%}section{margin:25px 0;padding:15px;background:#f0f4f7}.note{background:#fff0cd;padding:14px}</style><h1>原轨迹与鲁棒曲线平滑：同窗转速、方向对照</h1><p>保留原 YOLO/光流关联与至少三帧筛选；在同一个连续段内，按实际微秒时间用前后邻域拟合局部三次曲线。Huber 迭代降低异常中心偏移影响，框形变明显或光流样本权重较低。7 帧以下不追加平滑；不补帧、不跨 ID 或缺口，不强制坐标单调。</p><p class="note">平滑前后保持球半径、相机相对方向与拟合窗口不变。更光滑不等于物理测量更准确；原有框中心已经过 Kalman，叠加平滑仍可能偏移或削弱运动，当前保留双版本供审核。方向是相机局部前表面运动。</p><p>角度计算：先去除球体平移并归一化球半径，得到 X=(u−cx)/R、Y=(v−cy)/R、Z=−√(1−X²−Y²)。多点共同拟合 s(t+Δt)=Exp([ω]×Δt)s(t)，再由 |ω|/(2π) 得转/秒，ω/|ω| 得有向轴。单个点的球心夹角不一定等于球转角，靠近转轴的点移动尤其小。</p><p><a href="smoothing_report.json">完整平滑诊断</a> · <a href="smoothed_rotation_tracks.csv">平滑坐标与原坐标 CSV</a> · <a href="matched_spin_windows.csv">固定窗口转速/方向对照</a> · <a href="segment_metrics.csv">逐段轨迹指标</a> · <a href="smoothing_review_bundle.zip">下载审核包</a></p>'''
    page+=f"<p>共 {summary['observations']} 点、{summary['segments']} 段；{summary['smoothed_observations']} 点追加平滑。中心修正中位数 {summary['median_correction_px']:.3f} px，最大 {summary['maximum_correction_px']:.3f} px。局部反向步诊断 {summary['raw_local_reverse_steps']} → {summary['smoothed_local_reverse_steps']}（工程现象计数，不代表真实错点数）。</p>"
    for item in overlays:
        n=item['serve'];page+=f'<section><h2>{n}</h2>'
        page+=f'<video controls preload="metadata" poster="posters/{n}.png" src="videos/{n}_smoothing.mp4"></video>' if videos else f'<img src="posters/{n}.png">'
        page+='<p>同时间、同 ID 对比；右侧灰线为原路径，绿色为平滑后的中心与路径。黑侧只是未导出该时刻裁剪，完整 RAW 见之前的审核。</p>'
        for c in ['F3E','F59']:page+=f'<img src="plots/{n}_{c}.png">'
        page+='</section>'
    (output/'review.html').write_text(page+'</html>',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks',type=Path,default=Path('output/latest/tracking/rotation_tracks_all.csv'))
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--config',type=Path,default=Path('config/offline_marker_spin.json'))
    p.add_argument('--smoothing-config',type=Path,default=Path('config/marker_trajectory_smoothing.json'))
    p.add_argument('--baseline',type=Path,default=Path('output/latest_radius/r100/serve_spin_report.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--no-videos',action='store_true')
    a=p.parse_args();s=build(a.tracks,a.dataset,json.loads(a.config.read_text()),json.loads(a.smoothing_config.read_text()),a.baseline,a.output,not a.no_videos)
    from finalize_track_smoothing import finalize
    finalize(a.output,a.tracks,a.dataset)
    print(json.dumps({k:v for k,v in s.items() if k!='source_diagnostics'},ensure_ascii=False,indent=2))
