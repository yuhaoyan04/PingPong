"""Three retained results, independent camera rates, shared rate, local directions."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pingpong_event_pipeline.marker_spin import read_tracks,sha256,make_pairs
from pingpong_event_pipeline.spin import _rotate_vectors
from pingpong_event_pipeline.raw_sync import load_serves
from pingpong_event_pipeline.radius_review import font,CachedDraw
from pingpong_event_pipeline.clip_sync import ffmpeg_writer,finish
from pingpong_event_pipeline.motion import write_csv
from pingpong_event_pipeline.motion_review import color
from pingpong_event_pipeline.spin_fusion import frame_spin,time_window_id,marker_motion,check_motion_pair
from pingpong_event_pipeline.serve_spin import analyze_serves
from refine_observations import load,dump

MODES={'baseline':'原始结果','geometry_only':'球心与半径优化','combined':'组合优化＋追踪续接'}
SOURCES={'JOINT':'双视角联合','F3E':'F3E 单侧回退','F59':'F59 单侧回退',None:'暂无可靠结果'}


def prepare(root,source,output):
    variants={v['mode']:v for v in load(source/'variants.json')}
    variants['baseline']={'dataset':str(root/'data/server_marker_two_camera_train_v1'),
                         'tracks':str(root/'output/latest_smoothing/smoothed_rotation_tracks.csv')}
    for sub in ['tracks','spin','videos','posters','plots']:(output/sub).mkdir(parents=True,exist_ok=True)
    kept=[]
    baseline=load(root/'output/latest_smoothing/after_spin.json')
    fixed={s['serve']:{k:s[k] for k in ['R_F59_to_F3E','orientation_training_serves']} for s in baseline['serves']}
    for mode in MODES:
        v=variants[mode];target=output/'tracks'/f'{mode}.csv'
        if not target.exists():shutil.copy2(v['tracks'],target)
        assert sha256(target)==sha256(v['tracks'])
        src=source/'spin'/f'{mode}.json';dst=output/'spin'/f'{mode}.json'
        if src.exists():
            if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
        elif mode=='baseline':dump(dst,baseline)
        else:dump(dst,analyze_serves(target,v['dataset'],baseline['config'],fixed_relative_orientations=fixed))
        assert load(dst)['tracks_sha256']==sha256(target),'spin report / tracks mismatch'
        kept.append({'mode':mode,'label':MODES[mode],'dataset':v['dataset'],'tracks':str(target.resolve()),'tracks_sha256':sha256(target)})
    dump(output/'variants.json',kept)
    return {v['mode']:v for v in kept}


def direction_audit(output,variants):
    """Correlated fit diagnostics, not an independent accuracy measurement."""
    results=[]
    def angle(a,b):
        denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/denominator,-1,1)))
    for mode,v in variants.items():
        report=load(output/'spin'/f'{mode}.json')
        segments,_,_=read_tracks(v['tracks'],v['dataset']);values=[]
        for serve in report['serves']:
            for window in serve['windows']:
                for camera,fit in window['mono'].items():
                    if not fit.get('numerically_usable') or fit.get('rotation_signal')=='near_zero_unresolved_direction':continue
                    runs=[(key[2],run) for key,run in segments.items() if key[:2]==(camera,f"{serve['serve']}_{camera}")]
                    pairs=make_pairs(runs,window['start_us'],window['end_us'],report['config'].get('maximum_pair_us',2000))
                    if not pairs:continue
                    a=np.array([p[1]['sphere'] for p in pairs]);b=np.array([p[2]['sphere'] for p in pairs]);dt=np.array([p[3] for p in pairs])
                    w=np.array(fit['omega_camera_rad_s']);forward=_rotate_vectors(a,w,dt);reverse=_rotate_vectors(a,-w,dt)
                    observed=b[:,:2]-a[:,:2];local=forward[:,:2]-a[:,:2];proxy=np.tile([-w[1],w[0]],(len(pairs),1))
                    mask=(np.arccos(np.clip(np.sum(forward*b,axis=1),-1,1))<=report['config'].get('angular_threshold_rad',.035))
                    mask&=(np.linalg.norm(observed,axis=1)>1e-9)&(np.linalg.norm(local,axis=1)>1e-9)&(np.linalg.norm(proxy,axis=1)>1e-9)
                    if not mask.any():continue
                    radii=np.array([p[1]['radius'] for p in pairs])
                    values.extend(zip(angle(proxy[mask],observed[mask]),angle(local[mask],observed[mask]),
                        (np.linalg.norm(forward[:,:2]-b[:,:2],axis=1)*radii)[mask],
                        (np.linalg.norm(reverse[:,:2]-b[:,:2],axis=1)*radii)[mask]))
        medians=np.median(values,axis=0).tolist() if values else [None]*4
        results.append(dict(zip(['mode','correlated_inlier_pairs','old_front_proxy_median_direction_error_degrees',
            'local_finite_rotation_median_direction_error_degrees','forward_median_pixel_error','sign_flipped_median_pixel_error'],
            [mode,len(values),*medians]),scope='Same fitted windows and inlier observations; diagnostic, not held-out physical truth.'))
    dump(output/'direction_audit.json',results)


def arrow(draw,start,delta,fill,width=2):
    vector=np.asarray(delta,float);norm=np.linalg.norm(vector)
    if norm<1e-9:return
    unit=vector/norm;end=np.asarray(start)+vector
    draw.line([tuple(start),tuple(end)],fill=fill,width=width)
    base=end-unit*5;perp=np.array([-unit[1],unit[0]])*3
    draw.polygon([tuple(end),tuple(base+perp),tuple(base-perp)],fill=fill)


def view_sense(draw,xy,direction):
    x,y=xy;sense=direction.get('image_z_spin_sense') if direction else 'unresolved'
    text={'clockwise':'视轴分量：顺时针','counterclockwise':'视轴分量：逆时针',
          'small_view_axis_component':'视轴分量较弱','unresolved':'方向未分辨'}[sense]
    draw.text((x,y),text,font=font(16),fill='#cbd8e6')


def marker_arrows(row,run,independent,system_local,system_resolved):
    details=[]
    if row['track_support']!='ring_supported':return details
    index=next(i for i,r in enumerate(run) if r['timestamp']==row['timestamp'])
    neighbor=run[index+1] if index+1<len(run) else run[index-1] if index else None
    if neighbor is None or neighbor['track_support']!='ring_supported':return details
    a,b=(row,neighbor) if row['timestamp']<neighbor['timestamp'] else (neighbor,row)
    dt=(b['timestamp']-a['timestamp'])*1e-6
    choices=[]
    if independent['available'] and independent['direction_resolved']:
        choices.append(('independent',independent['omega_camera_rad_s'],'#43e5ef'))
    if system_local and system_resolved:
        choices.append(('system',system_local['omega_camera_rad_s'],'#ffb857'))
    for source,w,col in choices:
        check=check_motion_pair(a['sphere'],b['sphere'],dt,w,row['radius'])
        motion=marker_motion(row['sphere'],w,row['radius'],500)
        details.append({'model':source,'color':col,'motion':motion,**check})
    return details


def render(root,output,variants):
    dataset=root/'data/server_marker_two_camera_train_v1'
    reports={mode:{s['serve']:s for s in load(output/'spin'/f'{mode}.json')['serves']} for mode in MODES}
    points={}
    for mode,v in variants.items():
        segments,_,_=read_tracks(v['tracks'],v['dataset']);points[mode]=defaultdict(list)
        for (_,group,_),run in segments.items():
            for r in run:points[mode][(group,r['timestamp'])].append((r,run))
    overlays=[];flat=[];direction_rows=[];validated=[]
    for spec in load_serves(dataset):
        name=spec['serve'];maps={c:{int(r['timestamp_us']):r for r in spec['cameras'][c]['rows']} for c in ['F3E','F59']}
        times=sorted(set(maps['F3E'])|set(maps['F59']));values=[];model_arrows=[]
        for t in times:
            wid=time_window_id(reports['baseline'][name],t)
            data={mode:frame_spin(reports[mode][name],t,wid) for mode in MODES};values.append(data)
            per_frame={}
            for mode,value in data.items():
                s=value['system'];flat.append({'serve':name,'mode':mode,'timestamp_us':t,'window_id':wid,
                    'F3E_rps':value['independent']['F3E']['rps'],'F59_rps':value['independent']['F59']['rps'],
                    'F3E_window_id':value['independent']['F3E'].get('window_id'),
                    'F59_window_id':value['independent']['F59'].get('window_id'),'system_window_id':s.get('window_id'),
                    'system_rps':s['rps'],'system_source':s.get('source'),'system_selection':s.get('selection'),
                    'warnings':';'.join(value['warnings'])})
                per_frame[mode]={}
                for camera in ['F3E','F59']:
                    items=[]
                    for row,run in points[mode][(f'{name}_{camera}',t)]:
                        arrows=marker_arrows(row,run,value['independent'][camera],value['system_camera_directions'].get(camera),s.get('direction_resolved',False))
                        for a in arrows:
                            direction_rows.append({'serve':name,'mode':mode,'camera':camera,'timestamp_us':t,'track_id':row['track_id'],
                                'model':a['model'],'accepted':a['accepted'] and a['motion']['resolved'],
                                'reason':a['reason'] or (None if a['motion']['resolved'] else 'projection_unresolved'),
                                'direction_error_degrees':a['direction_error_degrees'],'angular_error_rad':a['angular_error_rad'],
                                'actual_pair_dt_us':a['actual_pair_dt_us']})
                        items.append((row,run,arrows))
                    per_frame[mode][camera]=items
            model_arrows.append(per_frame)
        def draw_frame(i):
            t=times[i];canvas=Image.new('RGB',(1440,1280),'#142130');draw=CachedDraw(canvas)
            draw.text((14,8),f'{name} | 源时间 {t} μs | 同一源时间，允许有效重叠窗口覆盖',font=font(24),fill='white')
            draw.text((14,48),'轨迹：真实观测；青箭头：本视角模型；橙箭头：系统模型。箭头按标记所在位置投影，长度适当放大。',font=font(18),fill='#d1dfeb')
            for k,mode in enumerate(MODES):
                x=12+k*480;v=values[i][mode];s=v['system']
                draw.text((x,88),MODES[mode],font=font(23),fill='#efbb7c' if mode=='baseline' else '#76e8cc')
                rate=f"系统总转速 {s['rps']:.2f} 转/秒" if s['available'] else '系统总转速 N/A'
                draw.text((x,128),rate,font=font(27),fill='#efbb7c' if mode=='baseline' else '#76e8cc')
                label=SOURCES[s.get('source')]
                if s.get('reason')=='conflicting_independent_estimates':label='两侧独立结果冲突，系统暂不输出'
                elif v['warnings']:label+='；单侧候选存在分歧'
                draw.text((x,172),label,font=font(17),fill='#efd18d')
                for j,camera in enumerate(['F3E','F59']):
                    y=255+j*455;mono=v['independent'][camera];r=maps[camera].get(t)
                    headline=f"{camera} 独立转速 {mono['rps']:.2f} 转/秒" if mono['available'] else f'{camera} 独立转速 N/A'
                    draw.text((x,y-41),headline,font=font(23),fill='white')
                    if r:
                        folder=Path(variants[mode]['dataset'])/'unlabeled_inference'/camera/f'{name}_{camera}'
                        with Image.open(folder/r['image']) as im:picture=im.convert('RGB').resize((320,320))
                        pd=CachedDraw(picture)
                        for row,run,arrows in model_arrows[i][mode][camera]:
                            col=color(row['track_id']);tail=[tuple(p['point']*1.25) for p in run if 0<=t-p['timestamp']<=4000]
                            if len(tail)>1:pd.line(tail,fill=col,width=2)
                            location=row['point']*1.25
                            for a in reversed(arrows):
                                if not a['accepted'] or not a['motion']['resolved']:continue
                                delta=np.array(a['motion']['predicted_displacement_px']);norm=np.linalg.norm(delta)
                                delta=delta/max(norm,1e-9)*min(28,max(12,norm*1.25))
                                arrow(pd,location,delta,a['color'],3 if a['model']=='system' else 2)
                            px,py=location;pd.ellipse((px-3,py-3,px+3,py+3),fill=col)
                            pd.text((px+5,py-14),str(row['track_id']),font=font(13),fill=col)
                        canvas.paste(picture,(x,y))
                    else:
                        draw.rectangle((x,y,x+320,y+320),fill='#080e16');draw.text((x+15,y+140),'此源时间无导出裁剪',font=font(20),fill='#efd18d')
                    direction=mono['direction']
                    draw.text((x+337,y+36),'青：本侧',font=font(17),fill='#43e5ef')
                    draw.text((x+337,y+75),'橙：系统',font=font(17),fill='#ffb857')
                    draw.text((x+337,y+120),'0.5 ms',font=font(17),fill='#d1dfeb')
                    draw.text((x+337,y+150),'前向投影',font=font(16),fill='#d1dfeb')
                    draw.text((x+337,y+200),'方向矛盾',font=font(15),fill='#efd18d')
                    draw.text((x+337,y+226),'则隐藏箭头',font=font(15),fill='#efd18d')
                    axis=mono.get('axis_camera')
                    text='本侧有向轴 '+','.join(f'{a:+.2f}' for a in axis)+f' [{camera}]' if axis else '本侧有向轴：未分辨'
                    draw.text((x,y+333),text,font=font(17),fill='#d1dfeb')
                    view_sense(draw,(x,y+360),direction)
                    here=points[mode][(f'{name}_{camera}',t)]
                    if mono['available']:
                        reason='有效邻窗覆盖当前观测' if mono.get('overlap_window_fallback') else '当前观测参与旋转拟合'
                    elif not r:reason='当前源时间没有裁剪帧'
                    elif not here:reason='无连续三帧的可用标记观测'
                    elif all(row['track_support']!='ring_supported' for row,_ in here):reason='球边界不足；二维轨迹仍保留'
                    else:
                        rejects=mono.get('rejection_reasons',[])
                        reason='支持点或时长不足' if 'insufficient_observations' in rejects or 'insufficient_time_span' in rejects else '旋转一致性或几何条件不足'
                    draw.text((x,y+386),reason,font=font(16),fill='#a5dece' if mono['available'] else '#efd18d')
            draw.text((14,1190),'独立转速各自由该视角求解；总转速优先使用共享旋转模型，单侧回退明确标记，不取两侧数值的简单平均。',font=font(19),fill='#efd18d')
            draw.text((14,1231),'源步长 500 μs，25 FPS 慢放；缺少当前观测不沿用旧值。方向为相机坐标，双机物理同步与空间标定仍待验证。',font=font(18),fill='#d1dfeb')
            return canvas
        poster=max(range(len(times)),key=lambda i:(sum(v['system']['available'] for v in values[i].values()),
                   times[i] in maps['F3E'] and times[i] in maps['F59'],-abs(i-len(times)/2)))
        draw_frame(poster).save(output/'posters'/f'{name}.jpg',quality=94)
        path=output/'videos'/f'{name}_spin_direction.mp4'
        if not path.exists():
            process,log=ffmpeg_writer(path,1440,1280,25)
            for i in range(len(times)):process.stdin.write(np.asarray(draw_frame(i)).tobytes())
            finish(process,log)
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,img=cap.read()
            if not ok:break
            assert img.shape==(1280,1440,3);count+=1
        cap.release();assert count==len(times)
        validated.append({'serve':name,'video':str(path.relative_to(output)),'frames':count,'all_frames_decoded':True})
        overlays.append({'serve':name,'timestamps_us':times,'values':values})
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
        data=[r for r in flat if r['serve']==name]
        for ax,mode in zip(axes,MODES):
            rr=[r for r in data if r['mode']==mode]
            for field,col in [('F3E_rps','tab:blue'),('F59_rps','tab:orange'),('system_rps','teal')]:
                ax.plot([(r['timestamp_us']-times[0])/1000 for r in rr],[r[field] if r[field] is not None else np.nan for r in rr],label=field,color=col,linewidth=1.4)
            ax.set(title=mode,ylabel='RPS',ylim=(0,None));ax.grid(alpha=.2);ax.legend()
        axes[-1].set_xlabel('Actual source time since first crop (ms)');fig.savefig(output/'plots'/f'{name}_rates.png',dpi=150);plt.close(fig)
        print('Rendered / decoded',name,count,flush=True)
    dump(output/'frame_values.json',overlays);write_csv(output/'frame_rates.csv',flat);write_csv(output/'local_direction_checks.csv',direction_rows)
    dump(output/'video_validation.json',validated)
    summary={'retained_modes':list(MODES),'source_counts':dict(Counter((r['mode']+'/'+str(r['system_source'])) for r in flat)),
             'local_arrows_checked':len(direction_rows),'local_arrows_accepted':sum(r['accepted'] for r in direction_rows),
             'suppression_reasons':dict(Counter(r['reason'] for r in direction_rows if not r['accepted'])),
             'physical_accuracy_validated':False}
    dump(output/'presentation_summary.json',summary)
    return validated


def page(output,videos):
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>三组结果：独立转速、总转速与局部方向</title><style>body{max-width:1450px;margin:24px auto;padding:0 16px;font:16px system-ui;line-height:1.7;color:#203244}section{background:#f0f4f8;padding:18px;margin:24px 0}video,img{max-width:100%}button{font:inherit;padding:7px;margin:5px}.note{background:#fff0ce;padding:14px}summary{cursor:pointer}</style><h1>原始、球几何优化、组合优化与续接</h1><p>只展示三组结果。每列顶部显示系统总转速，两行分别显示 F3E、F59 独立转速；总转速标明双视角联合或单侧回退。三列对齐同一源时间，优先使用同一锚定窗口；若该窗口失败，允许仍覆盖当前真实观测的有效重叠窗口，绝不按转速高低选窗。各侧和系统实际使用的窗口 ID 均写入 CSV。</p><p class="note">组合版新增有限时长的待确认轨迹、带运动初值的光流及最近真实图像恢复；仍需两个高置信命中才确认，至少连续三帧真实检测或核验光流才导出。预测保持帧不参与转速。缺少转速时，画面会说明球边界不足、连续观测不足或拟合条件不足。</p><p>青色箭头为该相机独立模型，橙色为系统模型经相机坐标变换后的局部投影。与相邻真实观测明显矛盾、运动太小或几何不可靠时隐藏箭头；箭头长度适当放大，0.5 ms 前向投影不是新增观测。</p><p><a href="spin_review_bundle.zip">下载审核包</a> · <a href="REPORT.md">检查、修复与保留范围</a> · <a href="frame_rates.csv">逐帧独立与系统转速</a> · <a href="local_direction_checks.csv">逐点方向检查</a> · <a href="direction_audit.json">方向符号审计</a></p>'''
    for row in sorted(videos,key=lambda r:(r['serve']!='serve05',r['serve'])):
        n=row['serve'];text+=f'<section><h2>{n}</h2><video controls preload="none" poster="posters/{n}.jpg" src="{row["video"]}"></video><p><button onclick="step(this,-1)">前一帧</button><button onclick="step(this,1)">后一帧</button>500 μs 源步长，25 FPS 慢放。</p><details><summary>两视角与系统转速曲线</summary><img loading="lazy" src="plots/{n}_rates.png"></details></section>'
    text+='''<p>当前转速与有向轴仍为候选，方向为相机局部坐标；尚无独立物理同步与球桌坐标标定。无观测显示 N/A，有近零证据才显示近零转速。</p><script>function step(b,d){let v=b.closest('p').previousElementSibling;v.pause();v.currentTime=Math.max(0,Math.min(v.duration,v.currentTime+d/25));}</script></html>'''
    if (output/'serve05_diagnostics.json').exists():
        text=text.replace('<h1>', '<p><a href="CONTINUITY.md">serve05 缺失原因和追踪改进</a> · <a href="serve05_diagnostics.json">serve05 前后统计</a> · <a href="continuity_timeline.csv">覆盖时间线</a></p><h1>',1)
    (output/'review.html').write_text(text)


def package(output):
    path=output/'spin_review_bundle.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(output.rglob('*')):
            if not f.is_file() or f==path:continue
            relative=str(f.relative_to(output))
            if relative=='review.html':z.writestr(relative,f.read_text().replace('<a href="spin_review_bundle.zip">下载审核包</a>','<span>当前为已解压的审核包</span>'))
            else:z.write(f,relative)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('output/latest_observations'))
    p.add_argument('--output',type=Path,default=Path('output/spin_review_'+datetime.now().strftime('%Y%m%d_%H%M%S')));p.add_argument('--package-only',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if a.package_only:package(output);return
    variants=prepare(root,a.source.resolve(),output);direction_audit(output,variants)
    videos=render(root,output,variants);page(output,videos)
    shutil.copy2(root/'docs/SPIN_DIRECTION_REVIEW.md',output/'REPORT.md')
    if (output/'serve05_diagnostics.json').exists():shutil.copy2(root/'docs/TRACK_CONTINUITY.md',output/'CONTINUITY.md')
    package(output)


if __name__=='__main__':main()
