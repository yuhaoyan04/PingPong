"""One serve at a time: synchronized views, track tails and shared spin fits."""
from __future__ import annotations

import csv
import html
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .motion import write_csv
from .motion_review import color, font
from .spin_acceptance import STATUSES, LABELS
from .spin_direction import describe_omega

REASONS = {
    "insufficient_observations": "同窗有效观测不足", "low_inlier_fraction": "刚体旋转内点比例不足",
    "high_all_point_residual": "整体运动残差过高", "robust_fit_failed": "未找到足够刚体共识",
    "one_view_has_no_supported_correspondences": "该时窗至少一个视角没有可靠球面轨迹",
    "insufficient_simultaneous_observations": "双侧实际观测同时段不足",
    "inconsistent_two_view_surface_motion": "共享角速度不能充分解释两侧运动",
    "too_few_shared_pairs_in_one_view": "共同时间段内一侧点对不足",
    "insufficient_inliers_in_one_view": "一侧缺少联合模型内点",
    "high_joint_reprojection_error": "联合重投影误差过高",
}


def conclusion(serve):
    joint = serve["joint_summary"]
    if joint["engineering_stable"]:
        return {"mode": "experimental_two_view", "rps": joint["engineering_candidate_rps"],
                "direction": LABELS[joint["front_motion_label"]], "text": "实验性双视角联合候选"}
    stable = [(c, v["aligned_summary"]) for c, v in serve["views"].items() if v["aligned_summary"]["engineering_stable"]]
    if stable:
        rates = [s['engineering_candidate_rps'] for _, s in stable]
        if len(stable) > 1 and (max(rates)-min(rates))/max(float(np.mean(rates)), .5) > .20:
            return {"mode": "unresolved", "rps": None, "direction": "未确定", "text": "两侧单视角转速候选冲突，需复核"}
        if len(stable) > 1 and serve['R_F59_to_F3E'] is not None and all(s['axis_x'] is not None for _,s in stable):
            axes = {c: np.array([s[k] for k in ['axis_x','axis_y','axis_z']]) for c,s in stable}
            angle = float(np.degrees(np.arccos(np.clip(axes['F3E'] @ np.array(serve['R_F59_to_F3E']) @ axes['F59'], -1, 1))))
            if angle > 20:
                return {"mode": "unresolved", "rps": None, "direction": "未确定", "text": "两侧有向轴候选冲突，需复核"}
        c, summary = max(stable, key=lambda item:(item[1]['stable_windows'], serve['views'][item[0]]['supported_observed_frames']))
        return {"mode": "single_view_fallback", "camera": c, "rps": summary["engineering_candidate_rps"],
                "direction": LABELS[summary["front_motion_label"]], "text": f"{c} 单视角候选；联合结果不足",
                "selection_rule": "more_temporally_consistent_windows_then_supported_observed_frames",
                "single_view_alternatives": [{"camera":c,"rps":s['engineering_candidate_rps']} for c,s in stable]}
    return {"mode": "unresolved", "rps": None, "direction": "未确定", "text": "证据不足，保留原始候选供复核"}


def diagnostic_text(serve):
    messages = []
    for camera, view in serve["views"].items():
        previous = view.get("previous_summary", {})
        status = previous.get("engineering_direction_status")
        prefix = f'{camera}：原先 {previous.get("usable_windows", 0)}/{previous.get("windows", 0)} 个窗口数值可用'
        if status == "insufficient_windows":
            prefix += '，未满足原先至少 3 个窗口的稳定性规则'
        if status == "inconsistent_candidate":
            excluded = previous.get("excluded_windows", [])
            prefix += f'；{len(excluded)} 个窗口超出轴向 15° 或转速相对 20% 的共识阈值'
        prefix += f'。{view["rotation_observations"]} 个轨迹观测，覆盖 {view["observed_frames"]}/{view["frames"]} 帧；最长连续段 {view["longest_segment_frames"]} 帧。'
        span=view['supported_pair_span_us']
        if span:
            prefix+=f'可靠点对时段 {span[0]}–{span[1]} μs。'
        prefix+=f'统一时窗后 {view["aligned_usable_windows"]} 个窗口数值可用；窗口边界会影响离散门槛，状态变化不代表精度已提高。'
        if view["unsupported_ball_frames"]:
            prefix += f'{view["unsupported_ball_frames"]} 帧球轮廓不可靠，不能用于球面旋转点对。'
        counts = view.get("previous_rejections", view["aligned_rejections"])
        if counts:
            prefix += '拒绝记录：' + '；'.join(f'{REASONS.get(k,k)} {n} 窗' for k,n in counts.items()) + '。'
        messages.append(prefix)
    if serve["serve"] == "serve05":
        messages.append('两侧图像时段有重叠，但进入旋转拟合的可靠轨迹时段不重叠。不能把较早的 F3E 与较晚的 F59 直接拼成同一时刻测量；缺数据不能输出为 0。')
    if serve["serve"] == "serve04":
        messages.append('F3E 的失败由运动共识与残差触发；这提示关联、簇中心或球面几何需要复核，单凭现有数据不能断定具体是哪一种。联合模型尝试用 F59 的约束识别 F3E 中一致的子集。')
    if serve["serve"] == "serve07":
        messages.append('F59 后段转速候选升至约 29–30 RPS，而 F3E 已离开可靠观测区。原先按整组做稳定性检查会受影响；联合模型只在同时有支持的时段拟合，不强行延伸到后段。真实变化与追踪偏差尚不能仅凭拟合区分。')
    return messages


def plot_serve(serve, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)
    first = min(v["start_us"] for v in serve["views"].values())
    for camera, col in [("F3E", "tab:blue"), ("F59", "tab:orange")]:
        rows = [w for w in serve["windows"] if w["mono"][camera]["numerically_usable"]]
        axes[0].scatter([(w["start_us"]-first)/1000 for w in rows], [w["mono"][camera]["candidate_rps"] for w in rows], label=camera+' independent', color=col, s=27)
        for w in serve["windows"]:
            v = w.get("per_view", {}).get(camera)
            if v:
                axes[1].scatter((w["start_us"]-first)/1000, v["inlier_fraction"], color=col,
                                marker='o' if w["numerically_usable"] else 'x', s=26)
    rows = [w for w in serve["windows"] if w["numerically_usable"]]
    axes[0].scatter([(w["start_us"]-first)/1000 for w in rows], [w["candidate_rps"] for w in rows], marker='D', color='tab:green', label='Shared two-view omega', s=38)
    axes[0].set(ylabel='Spin candidate (rps)', title=serve["serve"]+' | same physical window for both views', ylim=(0,None))
    axes[0].legend(fontsize=9)
    axes[1].axhline(.7, color='.5', linestyle=':', label='0.70 reference')
    axes[1].set(xlabel='Window start since first source frame (ms)', ylabel='Per-view joint-model inlier fraction', ylim=(0,1.05))
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(path, dpi=135); plt.close(fig)


def paired_media(serve, tracks, dataset, output, videos=True):
    folders, frames, by_time, runs = {}, {}, {}, {}
    for camera in ("F3E", "F59"):
        folder = Path(dataset) / "unlabeled_inference" / camera / f'{serve["serve"]}_{camera}'
        folders[camera] = folder
        with (folder / 'manifest.csv').open(encoding='utf-8-sig') as stream:
            frames[camera] = {int(row['timestamp_us']): row for row in csv.DictReader(stream)}
        by_time[camera], runs[camera] = defaultdict(list), defaultdict(list)
        for row in tracks:
            if row['group'] == f'{serve["serve"]}_{camera}':
                by_time[camera][int(row['timestamp_us'])].append(row)
                runs[camera][row['segment_id']].append(row)
        for run in runs[camera].values(): run.sort(key=lambda row:int(row['timestamp_us']))
    times = sorted(set(frames['F3E']) | set(frames['F59']))
    first = times[0]
    representative = max(times, key=lambda t:(min(len(by_time[c].get(t,[])) for c in frames),sum(len(by_time[c].get(t,[])) for c in frames),-abs(t-(times[0]+times[-1])/2)))
    r = np.array(serve['R_F59_to_F3E']) if serve['R_F59_to_F3E'] is not None else None
    def render(t):
        canvas=Image.new('RGB',(768,464),'#101923'); draw=ImageDraw.Draw(canvas)
        draw.text((15,10),f'{serve["serve"]} | source t={t} us | +{(t-first)/1000:.1f} ms',font=font(16),fill='white')
        supported=[w for w in serve['windows'] if w['numerically_usable'] and w['common_support']['start_us'] <= t <= w['common_support']['end_us']]
        fit=min(supported,key=lambda w:abs((w['start_us']+w['end_us'])/2-t),default=None)
        for j,camera in enumerate(['F3E','F59']):
            x=24+j*384; row=frames[camera].get(t)
            if row:
                patch=Image.open(folders[camera]/row['image']).convert('RGB').resize((320,320))
                canvas.paste(patch,(x,65)); draw=ImageDraw.Draw(canvas)
                draw.ellipse((x+160-133.333,65+160-133.333,x+160+133.333,65+160+133.333),outline='#698197',width=1)
                for marker in by_time[camera].get(t,[]):
                    col=color(marker['track_id'])
                    tail=[p for p in runs[camera][marker['segment_id']] if 0<=t-int(p['timestamp_us'])<=4000]
                    points=[(x+float(p['center_x_px'])*1.25,65+float(p['center_y_px'])*1.25) for p in tail]
                    if len(points)>1: draw.line(points,fill=col,width=2)
                    px,py=x+float(marker['center_x_px'])*1.25,65+float(marker['center_y_px'])*1.25
                    draw.ellipse((px-4,py-4,px+4,py+4),outline=col,width=2)
                    draw.text((px+5,py-8),str(marker['track_id']),font=font(12),fill=col)
                detail=f'frame {row["frame_index"]} | observed markers {len(by_time[camera].get(t,[]))}'
                draw.text((x,390),detail,font=font(13),fill='#d8e7f4')
                draw.text((x,410),row['track_support'],font=font(13),fill='#a9bed1' if row['track_support']=='ring_supported' else '#ffaa80')
            else:
                draw.text((x+25,185),'No exported crop at this time',font=font(16),fill='#b3bdc7')
            label=camera
            if fit:
                omega=np.array(fit['omega_camera_rad_s'])
                if camera=='F59': omega=r.T@omega
                d=describe_omega(omega if fit.get('rotation_signal')!='near_zero_unresolved_direction' else [0,0,0])
                label+=' | front: '+d['front_motion_label']
            draw.text((x,42),label,font=font(15),fill='#82d9ac')
        bottom=f'JOINT CANDIDATE: {fit["candidate_rps"]:.2f} rps | experimental orientation' if fit else 'No supported joint window at this timestamp (not zero spin)'
        draw.text((15,440),bottom,font=font(14),fill='#82d9ac' if fit else '#e8c185')
        return canvas
    render(representative).save(output/'paired_frames'/f'{serve["serve"]}.png')
    if not videos:
        return {"source_frames":len(times), "video_generated":False}
    destination=output/'videos'/f'{serve["serve"]}.mp4'
    with (destination.with_suffix('.log')).open('w') as log:
        p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','768x464','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','21','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE,stderr=log)
        try:
            for t in times: p.stdin.write(np.asarray(render(t)).tobytes())
        finally: p.stdin.close()
        if p.wait()!=0: raise RuntimeError(f'ffmpeg failed: {destination.with_suffix(".log")}')
    return {"source_frames":len(times),"video_generated":True,"playback_fps":25,
            "source_timestamps_us":times,"representative_timestamp_us":representative}


def write_review(report, tracks_path, output, videos=True, reuse_media=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name in ['plots','paired_frames','videos']: (output/name).mkdir(exist_ok=True)
    from .marker_spin import read_tracks
    from .trajectory_smoothing import csv_rows
    effective,_,_=read_tracks(tracks_path,report['dataset'],smoothing_config=report['config'].get('trajectory_smoothing'))
    tracks=csv_rows(effective)
    write_csv(output/'effective_rotation_tracks.csv',tracks)
    summary_rows,window_rows=[],[]
    for serve in report['serves']:
        serve['conclusion']=conclusion(serve); serve['diagnostic_text']=diagnostic_text(serve)
        if not (reuse_media and 'media' in serve and (output/'paired_frames'/f'{serve["serve"]}.png').is_file()
                and (not videos or (output/'videos'/f'{serve["serve"]}.mp4').is_file())):
            serve['media']=paired_media(serve,tracks,report['dataset'],output,videos)
        plot_serve(serve,output/'plots'/f'{serve["serve"]}.png')
        result=serve['conclusion']; joint=serve['joint_summary']
        summary_rows.append(dict(serve=serve['serve'],mode=result['mode'],candidate_rps=result['rps'],candidate_rpm=result['rps']*60 if result['rps'] is not None else None,
            direction_camera=result.get('camera','F3E') if result['mode']!='unresolved' else None,front_motion=result['direction'],
            joint_usable_windows=serve['joint_usable_windows'],joint_status=joint['engineering_direction_status'],orientation_mode=serve['orientation_mode'],physical_accuracy_validated=False))
        for w in serve['windows']:
            row={k:w.get(k) for k in ['serve','start_us','end_us','numerically_usable','candidate_rps','raw_fit_rps','balanced_inlier_fraction','axis_x','axis_y','axis_z','front_motion_label']}
            row['rejection_reasons']=';'.join(w['rejection_reasons'])
            for camera in ['F3E','F59']:
                row[camera+'_independent_rps']=w['mono'][camera]['candidate_rps']
                row[camera+'_joint_inlier_fraction']=w.get('per_view',{}).get(camera,{}).get('inlier_fraction')
            window_rows.append(row)
    write_csv(output/'serve_summary.csv',summary_rows)
    write_csv(output/'joint_windows.csv',window_rows)
    (output/'serve_spin_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PingPong 逐次发球双视角审核</title><style>body{font:16px system-ui;max-width:1200px;margin:25px auto;padding:0 18px;color:#192332;background:#f4f6f9;line-height:1.6}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img,video{max-width:100%;display:block;margin:12px auto}.note{background:#fff0cd;padding:15px}.card{background:white;padding:18px;margin:22px 0;border-radius:8px}a{color:#185b99}select{font-size:18px;padding:6px}.hidden{display:none}</style><h1>逐次发球：双视角互补与共享转速模型</h1>
<p>转速允许从 0 开始。不转 / 几乎不转、没有足够观测、方向未分辨是三种不同情况；没有数据时不填 0。图表从 0 起画，文献范围不充当下限。</p>
<p class="note">当前为实验性联合候选：两个视角在相同源时间上拟合同一个有向角速度。相对相机方向由其它发球估计，每次目标发球都被排除；逐次留出检验通过仍不等于实测外参或测量精度已验证。缺失实测标定与对应发球 RAW 的限制继续保留。箭头表示相机画面中可见前表面的运动，不直接当作上旋 / 下旋。</p>
<p><a href="serve_summary.csv">逐次发球汇总 CSV</a> · <a href="joint_windows.csv">逐窗口与两侧内点 CSV</a> · <a href="serve_spin_report.json">完整诊断 JSON</a></p><table><tr><th>发球</th><th>当前可用候选</th><th>转/秒</th><th>方向所在视角 / 前表面运动</th><th>联合可用窗口 / 状态</th></tr>'''
    for row in summary_rows:
        serve=next(s for s in report['serves'] if s['serve']==row['serve'])
        rate='未确定' if row['candidate_rps'] is None else f'{row["candidate_rps"]:.2f}'
        page+=f'<tr><td>{row["serve"]}</td><td>{serve["conclusion"]["text"]}</td><td>{rate}</td><td>{row["direction_camera"] or "—"} / {row["front_motion"]}</td><td>{row["joint_usable_windows"]} / {STATUSES[row["joint_status"]]}</td></tr>'
    page+='</table>'
    registration=report['orientation_registration']
    held=[v for v in registration['leave_one_serve_out'] if v['angle_error_deg'] is not None]
    page+='<h2>选择一次发球</h2><select id="choice"><option value="all">显示全部</option>'
    page+=''.join(f'<option value="{s["serve"]}">{s["serve"]}</option>' for s in report['serves'])+'</select>'
    for serve in report['serves']:
        name=serve['serve']; page+=f'<section class="card" data-serve="{name}"><h2>{name}：{serve["conclusion"]["text"]}</h2>'
        page+=''.join('<p>'+html.escape(t)+'</p>' for t in serve['diagnostic_text'])
        if serve['media']['video_generated']:
            page+=f'<video controls preload="metadata" poster="paired_frames/{name}.png" src="videos/{name}.mp4"></video><p>双视角按导出时间戳对齐；25 FPS 为慢放。黑面板表示没有导出该时刻的球裁剪，不能推断原相机中没有球。彩色短线为同一连续段内已观测的轨迹，轨迹编号只在各相机内部有效。物理时钟同步另需原始数据核验。</p>'
        else: page+=f'<img src="paired_frames/{name}.png">'
        page+=f'<img src="plots/{name}.png"><p>双视角拒绝原因：'+html.escape('；'.join(f'{REASONS.get(k,k)} {n} 窗' for k,n in serve['joint_rejections'].items()))+'</p>'
        alternatives=serve['conclusion'].get('single_view_alternatives',[])
        if alternatives:
            page+='<p>单视角候选分别为：'+ '；'.join(f'{v["camera"]} {v["rps"]:.2f} RPS' for v in alternatives)+'。当前回退选择稳定窗口较多的视角，其次比较有可靠球轮廓的已观测帧数；这不是精度排名，也未把两个数平均成联合结果。</p>'
        page+='<table><tr><th>联合窗口起点 μs</th><th>RPS</th><th>F3E / F59 内点比例</th><th>前表面运动（F3E）</th><th>相对整次发球共识</th></tr>'
        excluded={w['window_id'] for w in serve['joint_summary']['excluded_windows']}
        for w in serve['windows']:
            if w['numerically_usable']:
                page+=f'<tr><td>{w["start_us"]}</td><td>{w["candidate_rps"]:.2f}</td><td>{w["per_view"]["F3E"]["inlier_fraction"]:.2f} / {w["per_view"]["F59"]["inlier_fraction"]:.2f}</td><td>{LABELS[w["front_motion_label"]]}</td><td>{"偏离，保留复核" if w["window_id"] in excluded else "参与候选共识"}</td></tr>'
        page+='</table></section>'
    page+='<h2>相对方向估计的逐次留出检查</h2><p>留出整次发球，再用其它发球拟合 F59 → F3E 的正旋转矩阵；有向角速度不允许翻转符号。</p><table><tr><th>留出发球</th><th>方向误差 °</th><th>参与估计的其它发球</th></tr>'
    for v in held: page+=f'<tr><td>{v["held_out_serve"]}</td><td>{v["angle_error_deg"]:.2f}</td><td>{", ".join(v["registration"]["training_serves"])}</td></tr>'
    page+='</table><p>稳定性使用至少 3 窗口、轴向 15°、转速相对 20%、60% 共识的工程规则；重叠窗口相关，不是独立精度实验。当前报告不把单视角回退结果伪装成双视角结果。</p><script>document.getElementById("choice").onchange=function(){document.querySelectorAll("section[data-serve]").forEach(s=>{s.classList.toggle("hidden",this.value!=="all"&&s.dataset.serve!==this.value);s.querySelectorAll("video").forEach(v=>v.pause())})}</script></html>'
    (output/'review.html').write_text(page,encoding='utf-8')
    return summary_rows
