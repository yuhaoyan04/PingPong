"""Four controlled observation variants: fit rotation, render, and package review."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pingpong_event_pipeline.marker_spin import read_tracks,sha256
from pingpong_event_pipeline.serve_spin import analyze_serves
from pingpong_event_pipeline.raw_sync import load_serves,IndexedEvents
from pingpong_event_pipeline.radius_review import baseline_source,current_fit,frame_value,font,CachedDraw,arrow
from pingpong_event_pipeline.clip_sync import ffmpeg_writer,finish
from pingpong_event_pipeline.motion_review import color
from pingpong_event_pipeline.motion import write_csv
from refine_observations import load,dump,rows,warp_to_reference


LABELS={'baseline':'原轨迹 / 原球几何','marker_only':'只改标记中心','geometry_only':'只改球心和半径','combined':'两步同时改进'}


def fit_variants(root,output):
    baseline=load(root/'output/latest_smoothing/after_spin.json')
    fixed={s['serve']:{k:s[k] for k in ['R_F59_to_F3E','orientation_training_serves']} for s in baseline['serves']}
    folder=output/'spin';folder.mkdir(exist_ok=True);dump(folder/'baseline.json',baseline)
    for variant in load(output/'variants.json'):
        path=folder/f"{variant['mode']}.json"
        if path.exists():
            assert load(path)['tracks_sha256']==sha256(variant['tracks']);continue
        report=analyze_serves(variant['tracks'],variant['dataset'],baseline['config'],fixed_relative_orientations=fixed)
        dump(path,report);print('Fitted',variant['mode'],flush=True)


def summarize(output):
    metrics={};geometry=[]
    for path in sorted((output/'geometry').glob('*.json')):
        payload=load(path)
        for i,r in enumerate(payload['frames']):
            if not r.get('final_accepted') or r.get('audit_before_median_px') is None or r.get('final_audit_median_px') is None:continue
            old=np.array(payload['original_circles'][i]);new=np.array(payload['effective_circles'][i])
            geometry.append({'camera':payload['camera'],'group':payload['group'],'timestamp_us':payload['timestamps_us'][i],
                'center_change_sensor_px':float(np.linalg.norm(new[:2]-old[:2])),'radius_change_sensor_px':float(new[2]-old[2]),
                'audit_before_px':r['audit_before_median_px'],'audit_after_px':r['final_audit_median_px']})
    write_csv(output/'geometry_metrics.csv',geometry)
    metrics['geometry']={'accepted_frames':sum(r['refined_geometry_frames'] for r in load(output/'group_summary.json')),
        'diagnostic_frames':len(geometry),'total_frames':sum(r['frames'] for r in load(output/'group_summary.json')),
        'median_center_change_sensor_px':float(np.median([r['center_change_sensor_px'] for r in geometry])),
        'median_radius_change_sensor_px':float(np.median([r['radius_change_sensor_px'] for r in geometry])),
        'untouched_arc_median_error_before_sensor_px':float(np.median([r['audit_before_px'] for r in geometry])),
        'untouched_arc_median_error_after_sensor_px':float(np.median([r['audit_after_px'] for r in geometry])),
        'physical_accuracy_validated':False}
    local=[]
    for mode in ['marker_only','combined']:
        all_records=[r for p in (output/'localization').glob(f'*_{mode}.json') for r in load(p)]
        accepted=[r for r in all_records if r['accepted']]
        local.extend([{'mode':mode,'group':r['group'],'timestamp_us':r['timestamp_us'],'segment_id':r['segment_id'],
                       'correction_crop_px':r['center_correction_px'],'audit_before_crop_px':r['audit_before_median_px'],
                       'audit_after_crop_px':r['audit_after_median_px']} for r in accepted])
        metrics[mode]={'accepted_observations':len(accepted),'observations':len(all_records),
            'status_counts':dict(Counter(r['status'] for r in all_records)),
            'median_center_correction_crop_px':float(np.median([r['center_correction_px'] for r in accepted])) if accepted else None,
            'untouched_corner_median_error_before_crop_px':float(np.median([r['audit_before_median_px'] for r in accepted])) if accepted else None,
            'untouched_corner_median_error_after_crop_px':float(np.median([r['audit_after_median_px'] for r in accepted])) if accepted else None}
    write_csv(output/'localization_metrics.csv',local)
    reports={mode:load(output/'spin'/f'{mode}.json') for mode in LABELS}
    maps={mode:{s['serve']:s for s in report['serves']} for mode,report in reports.items()}
    matched=[];serve_summary=[]
    for name,b in maps['baseline'].items():
        for mode in ['marker_only','geometry_only','combined']:
            a=maps[mode][name];joined=[]
            for wb,wa in zip(b['windows'],a['windows']):
                assert wb['window_id']==wa['window_id']
                for source in ['JOINT','F3E','F59']:
                    x=wb if source=='JOINT' else wb['mono'][source];y=wa if source=='JOINT' else wa['mono'][source]
                    delta=None
                    if x.get('candidate_axis_camera') is not None and y.get('candidate_axis_camera') is not None:
                        delta=float(np.degrees(np.arccos(np.clip(np.dot(x['candidate_axis_camera'],y['candidate_axis_camera']),-1,1))))
                    record={'serve':name,'mode':mode,'source':source,'window_id':wb['window_id'],
                            'baseline_usable':x['numerically_usable'],'candidate_usable':y['numerically_usable'],
                            'baseline_rps':x.get('candidate_rps'),'candidate_rps':y.get('candidate_rps'),
                            'signed_axis_change_degrees':delta,'candidate_rejections':';'.join(y['rejection_reasons'])}
                    matched.append(record)
                    if source=='JOINT' and x['numerically_usable'] and y['numerically_usable']:joined.append(record)
            serve_summary.append({'serve':name,'mode':mode,'baseline_joint_usable_windows':b['joint_usable_windows'],
                'candidate_joint_usable_windows':a['joint_usable_windows'],'matched_joint_windows':len(joined),
                'matched_baseline_median_rps':float(np.median([r['baseline_rps'] for r in joined])) if joined else None,
                'matched_candidate_median_rps':float(np.median([r['candidate_rps'] for r in joined])) if joined else None,
                'matched_median_axis_change_degrees':float(np.median([r['signed_axis_change_degrees'] for r in joined if r['signed_axis_change_degrees'] is not None])) if joined and any(r['signed_axis_change_degrees'] is not None for r in joined) else None})
    write_csv(output/'matched_spin_windows.csv',matched);write_csv(output/'serve_comparison.csv',serve_summary)
    metrics['serve_comparison']=serve_summary;metrics['production_default_changed']=False
    metrics['physical_spin_accuracy_validated']=False
    dump(output/'metrics.json',metrics)
    (output/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,mode in zip(axes,['marker_only','geometry_only','combined']):
        rr=[r for r in matched if r['mode']==mode and r['source']=='JOINT' and r['baseline_usable'] and r['candidate_usable']]
        ax.scatter([r['baseline_rps'] for r in rr],[r['candidate_rps'] for r in rr],s=25,color='teal',alpha=.7)
        ax.plot([0,65],[0,65],'k--',alpha=.3);ax.set(title=mode,xlabel='Baseline RPS',ylabel='Candidate RPS',xlim=(0,65),ylim=(0,65))
    fig.suptitle('Matched windows, fixed camera orientation; physical accuracy unknown')
    fig.savefig(output/'plots/spin_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].boxplot([[r['audit_before_px'] for r in geometry],[r['audit_after_px'] for r in geometry]],showfliers=False)
    axes[0].set_xticks([1,2],['Original circle','Refined circle'])
    axes[0].set(title='Untouched edge sectors (accepted corrections)',ylabel='Radial residual (sensor px)')
    marker=[r for r in local if r['mode']=='marker_only']
    axes[1].boxplot([[r['audit_before_crop_px'] for r in marker],[r['audit_after_crop_px'] for r in marker]],showfliers=False)
    axes[1].set_xticks([1,2],['BBox translation','Anchor affine'])
    axes[1].set(title='Untouched corners (accepted registrations)',ylabel='Feature prediction residual (crop px)')
    fig.savefig(output/'plots/observation_metrics.png',dpi=150);plt.close(fig)
    return metrics


def geometry_sheets(root,output,cache):
    (output/'geometry_sheets').mkdir(exist_ok=True)
    for path in sorted((output/'geometry').glob('*.json')):
        report=load(path);name=report['group'];camera=report['camera'];serve=name.rsplit('_',1)[0]
        source=IndexedEvents(np.load(cache/f'{serve}_{camera}.npy',mmap_mode='r'))
        times=np.array(report['timestamps_us'],np.int64);old=np.array(report['original_circles']);new=np.array(report['effective_circles'])
        accepted=[i for i,r in enumerate(report['frames']) if r.get('final_accepted')]
        picks=[accepted[round(x*(len(accepted)-1))] for x in [0,.2,.4,.6,.8,1]] if accepted else [round(x*(len(times)-1)) for x in [0,.2,.4,.6,.8,1]]
        canvas=Image.new('RGB',(1440,1100),'#142130');draw=CachedDraw(canvas)
        draw.text((15,10),f'{name} | 同一 RAW 事件窗，平移/缩放补偿到当前时刻 | 黄：原圆；青：新候选',font=font(23),fill='white')
        for k,i in enumerate(picks):
            t=times[i];x=20+(k%3)*480;y=85+(k//3)*500;seed=old[i]
            xy,_=warp_to_reference(source.window(t-1000,t),times,old,t)
            side=4*seed[2];scale=360/side
            uv=(xy-seed[:2])*scale+180
            valid=(uv[:,0]>=0)&(uv[:,0]<360)&(uv[:,1]>=0)&(uv[:,1]<360)
            image=np.zeros((360,360,3),np.uint8);u=uv[valid].astype(int);image[u[:,1],u[:,0]]=190
            canvas.paste(Image.fromarray(image),(x,y))
            for circle,col in [(seed,'#ffba54'),(new[i],'#67eddf')]:
                cx,cy=(circle[:2]-seed[:2])*scale+180+[x,y];r=circle[2]*scale
                draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=col,width=2)
            draw.text((x,y-30),f't={t} μs | 帧 {i}',font=font(18),fill='white')
            draw.text((x,y+365),f'R {seed[2]:.2f} → {new[i,2]:.2f} RAW px',font=font(19),fill='#67eddf')
            draw.text((x,y+395),f'球心修正 {np.linalg.norm(new[i,:2]-seed[:2]):.2f} RAW px',font=font(18),fill='#d1dfeb')
            state=report['frames'][i].get('continuous_status','原几何保留')
            draw.text((x,y+420),{'current_arcs_supported':'当前边界支持','short_gap_temporal_geometry':'邻帧约束的短缺口估计'}.get(state,'原几何保留'),font=font(15),fill='#efd18d')
        canvas.save(output/'geometry_sheets'/f'{name}.jpg',quality=94)


def render(root,output):
    dataset=root/'data/server_marker_two_camera_train_v1'
    variants={'baseline':{'tracks':str(root/'output/latest_smoothing/smoothed_rotation_tracks.csv'),'dataset':str(dataset)}}
    variants.update({v['mode']:v for v in load(output/'variants.json')})
    maps={mode:{s['serve']:s for s in load(output/'spin'/f'{mode}.json')['serves']} for mode in LABELS}
    bytime={}
    for mode in LABELS:
        segments,_,_=read_tracks(variants[mode]['tracks'],variants[mode]['dataset'])
        bytime[mode]=defaultdict(list)
        for key,run in segments.items():
            for r in run:bytime[mode][(key[1],r['timestamp'])].append((r,run))
    for sub in ['videos','posters']:(output/sub).mkdir(exist_ok=True)
    validation=[];overlays=[]
    for spec in load_serves(dataset):
        name=spec['serve'];frame_maps={c:{int(r['timestamp_us']):r for r in spec['cameras'][c]['rows']} for c in ['F3E','F59']}
        times=sorted(set(frame_maps['F3E'])|set(frame_maps['F59']));values=[]
        for t in times:
            source=baseline_source(maps['baseline'][name],t)
            fitted=current_fit(maps['baseline'][name],t,source) if source else None
            wid=fitted[0]['window_id'] if fitted else None
            values.append({mode:frame_value(maps[mode][name],t,source,wid) for mode in LABELS})
        def image(i):
            t=times[i];canvas=Image.new('RGB',(1440,1100),'#142130');draw=CachedDraw(canvas)
            draw.text((14,8),f'{name} | t={t} μs | 同源时间、同簇 ID、同旋转窗口与相机方向',font=font(24),fill='white')
            draw.text((14,48),'后两列使用 RAW 重新补偿的裁剪；新球心/半径由边界测量，未按预期转速调参。',font=font(20),fill='#efd18d')
            for k,mode in enumerate(LABELS):
                x=k*360+12;value=values[i][mode]
                draw.text((x,88),LABELS[mode],font=font(21),fill='#efbb7c' if mode=='baseline' else '#76e8cc')
                for j,camera in enumerate(['F3E','F59']):
                    y=165+j*435;row=frame_maps[camera].get(t);group=f'{name}_{camera}'
                    draw.text((x,y-33),camera,font=font(21),fill='white')
                    if row:
                        folder=Path(variants[mode]['dataset'])/'unlabeled_inference'/camera/group
                        with Image.open(folder/row['image']) as im:pic=im.convert('RGB')
                        canvas.paste(pic,(x,y))
                        rr=128/1.2;draw.ellipse((x+128-rr,y+128-rr,x+128+rr,y+128+rr),outline='#566f82',width=1)
                        for r,run in bytime[mode][(group,t)]:
                            col=color(r['track_id'])
                            trail=[(x+p['point'][0],y+p['point'][1]) for p in run if 0<=t-p['timestamp']<=4000]
                            if len(trail)>1:draw.line(trail,fill=col,width=2)
                            px,py=np.array([x,y])+r['point'];draw.ellipse((px-3,py-3,px+3,py+3),fill=col)
                            draw.text((px+5,py-13),str(r['track_id']),font=font(12),fill=col)
                    else:
                        draw.rectangle((x,y,x+256,y+256),fill='#080e16');draw.text((x+15,y+115),'此时刻无导出裁剪',font=font(19),fill='#efd18d')
                    arrow(draw,(x+300,y+128),value.get('directions',{}).get(camera),'#76e8cc')
                    rate=f"{value['rps']:.2f} 转/秒" if value['available'] else 'N/A'
                    draw.text((x,y+269),rate,font=font(25),fill='#efbb7c' if mode=='baseline' else '#76e8cc')
                    draw.text((x,y+309),f"来源 {value['source'] or '无当前有效观测'}",font=font(18),fill='#d1dfeb')
                    if value.get('axis'):
                        draw.text((x,y+345),'轴 '+','.join(f'{a:+.2f}' for a in value['axis'])+f" [{value['axis_camera']}]",font=font(16),fill='#d1dfeb')
            draw.text((14,1025),'500 μs 源步长，25 FPS 慢放；当前窗不合格即 N/A，不用其它窗口替换。箭头为可见前表面运动。',font=font(19),fill='#efd18d')
            draw.text((14,1064),'固定参考纹理中心仍非人工标定物理点；球几何与转速均为实验候选，两机物理同步仍待独立验证。',font=font(18),fill='#d1dfeb')
            return canvas
        poster=max(range(len(times)),key=lambda i:(sum(v['available'] for v in values[i].values()),times[i] in frame_maps['F3E'] and times[i] in frame_maps['F59'],values[i]['baseline']['source']=='JOINT',-abs(i-len(times)/2)))
        image(poster).save(output/'posters'/f'{name}.jpg',quality=94)
        path=output/'videos'/f'{name}_observation_refinement.mp4'
        if not path.exists():
            process,log=ffmpeg_writer(path,1440,1100,25)
            for i in range(len(times)):process.stdin.write(np.asarray(image(i)).tobytes())
            finish(process,log)
        capture=cv2.VideoCapture(str(path));count=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            assert frame.shape==(1100,1440,3);count+=1
        capture.release();assert count==len(times)
        validation.append({'serve':name,'video':str(path.relative_to(output)),'frames':count,'all_frames_decoded':True})
        overlays.append({'serve':name,'timestamps_us':times,'values':values});print('Rendered',name,count,flush=True)
    dump(output/'frame_values.json',overlays);dump(output/'video_validation.json',validation)
    return validation


def write_page(output,videos,metrics):
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>标记中心与球几何改进审核</title><style>body{font:16px system-ui;max-width:1450px;margin:25px auto;padding:0 14px;line-height:1.7;color:#213344}video,img{max-width:100%}section{background:#eff4f8;margin:25px 0;padding:18px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #c6d4e0}.note{padding:15px;background:#fff0cd}button{font:inherit;padding:7px;margin:5px}summary{cursor:pointer}</style><h1>形变下的标记中心定位、球心与半径估计</h1><p>四列依次为原版、只改标记中心、只改球几何、两步组合。保持相机方向与旋转窗口不变；后两列已从同一 RAW 事件窗口重新做运动补偿与裁剪。原始数据和生产模型保留。</p><p class="note">局部配准跟随同一参考纹理，能减少框扩散/收缩牵引；它仍不是人工标定的物理标记中心。球几何以 RAW 边界为约束，通过角度分组检验后才更新。诊断残差降低不等于已获得物理转速真值。</p><p><a href="observation_review_bundle.zip">下载审核包</a> · <a href="REPORT.md">算法、结果与限制</a> · <a href="metrics.json">全部指标</a> · <a href="matched_spin_windows.csv">同窗转速/方向对照</a> · <a href="serve_comparison.csv">逐发球汇总</a> · <a href="group_summary.json">球几何与中心定位分组统计</a></p><section><h2>观测层面的检查</h2>'''
    g=metrics['geometry'];m=metrics['marker_only']
    page+=f"<p>球几何：{g['total_frames']} 帧中 {g['accepted_frames']} 帧采用新测量；接受帧中未参与拟合或门控的边缘角度残差中位数 {g['untouched_arc_median_error_before_sensor_px']:.3f} → {g['untouched_arc_median_error_after_sensor_px']:.3f} RAW px。标记中心：{m['observations']} 点中 {m['accepted_observations']} 点采用新定位；对应诊断角点残差 {m['untouched_corner_median_error_before_crop_px']:.3f} → {m['untouched_corner_median_error_after_crop_px']:.3f} 裁剪 px。</p>"
    page+='''<p>指标取接受修正的样本；拟合、门控、诊断分别使用不同角度/角点，但仍来自同一事件观测，不能视为独立真值测试。几何短缺口用有邻近测量约束的连续模型，明确区分当前边界支持与邻帧估计；原有几何回退帧没有自动升级，缺观测也不会输出零转速。</p><p><a href="appearance/review.html">重建裁剪的纹理周期交叉检查</a>：不使用 YOLO ID 或转速拟合结果挑选周期峰，保留分数周/多周歧义。</p><img src="plots/observation_metrics.png"><img src="plots/spin_comparison.png"></section><section><h2>同步转速、方向与轨迹</h2>'''
    for r in videos:
        name=r['serve'];page+=f'<h3>{name}</h3><video controls preload="none" poster="posters/{name}.jpg" src="{r["video"]}"></video><p><button onclick="step(this,-1)">前一帧</button><button onclick="step(this,1)">后一帧</button>源步长 500 μs，25 FPS 慢放；可暂停放大检查同一 ID。</p>'
    page+='</section><section><h2>直接检查 RAW 球边界</h2><p>每图取六个分布在该组时间段的接受修正帧；若没有接受修正，则按时间均匀取帧。灰点是平移/缩放补偿到当前时刻的 RAW 事件，黄圈为原几何，青圈为新候选。未根据转速高低挑帧。</p>'
    for r in load(output/'group_summary.json'):
        page+=f'<details><summary>{r["group"]}：{r["refined_geometry_frames"]}/{r["frames"]} 帧更新几何</summary><img loading="lazy" src="geometry_sheets/{r["group"]}.jpg"></details>'
    page+='</section><script>function step(b,d){let v=b.closest("p").previousElementSibling;v.pause();v.currentTime=Math.max(0,Math.min(v.duration,v.currentTime+d/25));}</script></html>'
    (output/'review.html').write_text(page)


def package(output):
    path=output/'observation_review_bundle.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for source in sorted(output.rglob('*')):
            if not source.is_file() or source==path or 'dataset' in source.relative_to(output).parts:continue
            relative=str(source.relative_to(output))
            if relative=='review.html':z.writestr(relative,source.read_text().replace('<a href="observation_review_bundle.zip">下载审核包</a>','<span>当前为已解压审核包</span>'))
            else:z.write(source,relative)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('output/observation_refinement_v14'))
    p.add_argument('--cache',type=Path,default=Path('/tmp/pingpong_raw_sync_cache_v9'))
    p.add_argument('--fit-only',action='store_true');p.add_argument('--package-only',action='store_true')
    a=p.parse_args();output=a.output.resolve();root=Path(__file__).resolve().parents[1]
    if a.package_only:package(output);return
    fit_variants(root,output);metrics=summarize(output)
    if not a.fit_only:
        if not (output/'appearance/appearance_report.json').is_file():
            from review_appearance_period import build as appearance_review
            appearance_review(output/'dataset',output/'appearance',output/'spin/baseline.json')
        geometry_sheets(root,output,a.cache);videos=render(root,output);write_page(output,videos,metrics);package(output)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
