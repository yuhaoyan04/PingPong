"""Render controlled candidate comparisons and a portable whole-chain review."""
from __future__ import annotations
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
import shutil
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pingpong_event_pipeline.marker_spin import read_tracks
from pingpong_event_pipeline.raw_sync import load_serves
from pingpong_event_pipeline.radius_review import baseline_source,current_fit,frame_value,font,CachedDraw,arrow
from pingpong_event_pipeline.clip_sync import ffmpeg_writer,finish
from pingpong_event_pipeline.motion_review import color
from audit_system_iteration import load,dump


def render_videos(root,output):
    reports=[load(output/'rotation'/f'{m}.json') for m in ['uniform','cluster_equal']]
    mappings=[{s['serve']:s for s in r['serves']} for r in reports]
    dataset=root/'data/server_marker_two_camera_train_v1'
    segments,_,_=read_tracks(root/'output/latest_smoothing/smoothed_rotation_tracks.csv',dataset)
    bytime=defaultdict(list)
    for key,run in segments.items():
        for r in run:bytime[(key[0],key[1],r['timestamp'])].append((r,run))
    results=[];overlays=[]
    for spec in load_serves(dataset):
        name=spec['serve'];maps={c:{int(r['timestamp_us']):r for r in spec['cameras'][c]['rows']} for c in ['F3E','F59']}
        times=sorted(maps['F3E'].keys()|maps['F59'].keys());values=[]
        for t in times:
            source=baseline_source(mappings[0][name],t)
            fitted=current_fit(mappings[0][name],t,source) if source else None
            wid=fitted[0]['window_id'] if fitted else None
            values.append([frame_value(m[name],t,source,wid) for m in mappings])
        def render(i):
            t=times[i];canvas=Image.new('RGB',(1280,1000),'#142130');draw=CachedDraw(canvas)
            draw.text((16,10),f'{name} | 源时间 {t} μs | 固定半径、坐标、轨迹 ID、窗口与相机方向',font=font(22),fill='white')
            draw.text((16,49),'左：原点对权重；右：每个标记簇总权重均衡（实验选项，尚未替换默认算法）。',font=font(20),fill='#f5d19a')
            for j,c in enumerate(['F3E','F59']):
                row=maps[c].get(t);y=140+j*395
                for k,label in enumerate(['原点对权重','标记簇均衡']):
                    x=18+k*640;v=values[i][k]
                    draw.text((x,y-35),f'{c} | {label}',font=font(23),fill='#efb270' if k==0 else '#95e5c3')
                    if row:
                        with Image.open(Path(spec['cameras'][c]['folder'])/row['image']) as im:pic=im.convert('RGB').resize((320,320))
                        canvas.paste(pic,(x,y))
                        for r,run in bytime[(c,f'{name}_{c}',t)]:
                            trail=[(x+p['point'][0]*1.25,y+p['point'][1]*1.25) for p in run if 0<=t-p['timestamp']<=5000]
                            col=color(r['track_id'])
                            if len(trail)>1:draw.line(trail,fill=col,width=2)
                            px,py=x+r['point'][0]*1.25,y+r['point'][1]*1.25
                            draw.ellipse((px-4,py-4,px+4,py+4),fill=col)
                            draw.text((px+5,py-16),str(r['track_id']),font=font(14),fill=col)
                    else:
                        draw.rectangle((x,y,x+320,y+320),fill='#080e16');draw.text((x+25,y+135),'此时刻无导出裁剪',font=font(22),fill='#f5d19a')
                    rate=f"{v['rps']:.2f} 转/秒" if v['available'] else 'N/A'
                    draw.text((x+345,y+30),rate,font=font(27),fill='#efb270' if k==0 else '#95e5c3')
                    draw.text((x+345,y+77),str(v['source'] or '无当前有效观测'),font=font(19),fill='white')
                    arrow(draw,(x+445,y+172),v.get('directions',{}).get(c),'#95e5c3')
                    draw.text((x+345,y+238),'可见前表面运动方向',font=font(17),fill='#d1dfeb')
                    axis=v.get('axis')
                    if axis:draw.text((x,y+329),'有向轴 '+', '.join(f'{a:+.3f}' for a in axis)+f" [{v['axis_camera']}]",font=font(17),fill='#d1dfeb')
            draw.text((16,930),'只在当前真实观测时刻显示同窗候选；不保留旧值。N/A 表示约束不足或未通过检查。',font=font(19),fill='#f5d19a')
            draw.text((16,965),'源步长 500 μs，25 FPS 慢放。数字与方向均未完成独立物理标定；两机物理同步仍待验证。',font=font(18),fill='#d1dfeb')
            return canvas
        poster=max(range(len(times)),key=lambda i:(sum(v['available'] for v in values[i]),
                   times[i] in maps['F3E'] and times[i] in maps['F59'],values[i][0]['source']=='JOINT',-abs(i-len(times)/2)))
        render(poster).save(output/'posters'/f'{name}.jpg',quality=94)
        path=output/'videos'/f'{name}_cluster_weighting.mp4'
        if not path.exists():
            proc,log=ffmpeg_writer(path,1280,1000,25)
            for i in range(len(times)):proc.stdin.write(np.asarray(render(i)).tobytes())
            finish(proc,log)
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(1000,1280,3);count+=1
        cap.release();assert count==len(times),(name,count,len(times))
        results.append({'serve':name,'frames':count,'first_timestamp_us':times[0],'last_timestamp_us':times[-1],
                        'video':str(path.relative_to(output)),'all_frames_decoded':True})
        overlays.append({'serve':name,'timestamps_us':times,'values':values})
        print('Rendered / decoded',name,count,flush=True)
    dump(output/'frame_values.json',overlays)
    dump(output/'video_validation.json',results)
    return results


def training_plots(output):
    records=[]
    for recipe in ['holdout','augmented_holdout']:
        for camera in ['F3E','F59']:
            folds=load(output/recipe/camera/'holdout_summary.json')
            if len(folds)!=2:raise RuntimeError(f'Both training folds must finish before rendering {recipe}/{camera}')
            for fold in folds:
                for metric in fold['metrics']:
                    records.append({'recipe':recipe,'camera':camera,'heldout_serve':fold['heldout_serve'],**metric})
    dump(output/'training_comparison.json',records)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    labels=['F3E/01','F3E/06','F59/01','F59/06'];x=np.arange(4)
    for j,key in enumerate(['precision','recall']):
        for i,recipe in enumerate(['holdout','augmented_holdout']):
            selected=[r for r in records if r['recipe']==recipe and r['confidence']==.75]
            selected.sort(key=lambda r:(r['camera'],r['heldout_serve']))
            axes[j].bar(x+(i-.5)*.32,[r[key] for r in selected],width=.32,label='Original' if i==0 else 'Mild geometric aug.')
        axes[j].set(xticks=x,xticklabels=labels,ylim=(0,1.06),title=key+' | conf=0.75, IoU=0.50',xlabel='Camera / held-out serve');axes[j].legend()
    fig.suptitle('Within-recording two-fold diagnostics; production all-label models unchanged')
    fig.savefig(output/'plots/training_comparison.png',dpi=150);plt.close(fig)
    return records


def page(output,records,videos):
    audit=load(output/'dataset_audit.json');rotation=load(output/'rotation/summary.json');queue=load(output/'review_queue.json')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PingPong 全链路迭代审核</title><style>body{max-width:1280px;margin:24px auto;padding:0 16px;font:16px system-ui;line-height:1.65;color:#203244}section{background:#f0f4f8;padding:18px;margin:24px 0;border-radius:12px}video,img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #c8d3dd;text-align:left}.note{padding:14px;background:#fff0ce}button,select{font:inherit;padding:7px;margin:5px}summary{cursor:pointer}a{color:#12638d}</style>
    <h1>PingPong 全链路检查与迭代 v11</h1><p>已完成数据包、去噪坐标与配置、时间窗口、训练分组、追踪连续性、轨迹平滑及多点/双视角转速拟合检查。下方包含固定条件对照视频、诊断表与人工审核队列。</p>
    <p class="note">这是工程迭代，不是物理精度验收。仅有第 01、06 次发球标注；本次训练是同一次录制中的整次发球互留诊断。生产全量训练模型未替换；双机物理同步、实测相机标定、独立转速真值仍未完成。</p>
    <p><a href="iteration_review_bundle.zip">下载本轮审核包</a> · <a href="audit_report.md">完整检查结论与下一轮优先级</a> · <a href="dataset_audit.json">数据审计</a> · <a href="training_comparison.json">训练对照数据</a> · <a href="rotation/summary.json">转速对照</a></p>
    <section><h2>本轮已落地</h2><ul><li>RAW 保持原始事件顺序；球体处理不再排序掩盖时间倒退。</li><li>事件窗口统一半开区间，保留下一个窗口的边界事件；修复负时间修正的整数风险，异常批次明确报错。</li><li>新增可选的标记簇权重均衡，保持每个相机总权重相同；保留零转速与观测不足的区别。</li><li>原训练配方与温和几何增强分别完成四组固定 150 轮实验；不使用留出帧选 checkpoint。</li></ul></section>
    <section><h2>数据、去噪与球几何</h2><table><tr><th>相机</th><th>标注帧/框</th><th>确认空帧</th><th>推理帧</th><th>球几何回退</th><th>像素保留中位数</th></tr>'''
    for r in audit:
        content+=f"<tr><td>{r['camera']}</td><td>{r['reviewed_frames']} / {r['reviewed_boxes']}</td><td>{r['explicit_empty_negative_frames']}</td><td>{r['unlabeled_frames']}</td><td>{r['fallback_geometry_frames']}</td><td>{r['filter_pixel_retention_quantiles']['0.5']:.1%}</td></tr>"
    content+='''</table><p>全部 1,946 张图的当前去噪配置逐像素复现；输出均为原占据像素子集，极性占用位置一致。删除比例不是噪声识别率。回退球几何的观测不参与有效转速点对。</p></section><section><h2>训练对照</h2><p>同相机 train01→test06、train06→test01；固定最终 EMA，无留出集选模。温和增强：平移 ±5%、缩放 ±10%、水平翻转 50%；只用于检测器，不修改转速输入。相同录制内的结果仍有场景相关性，且这些诊断集已被查看，不能再称独立最终测试。</p><img src="plots/training_comparison.png"><table><tr><th>配方</th><th>相机/留出</th><th>阈值</th><th>TP / FP / FN</th><th>精确率</th><th>召回率</th><th>匹配框中心中位误差</th></tr>'''
    for r in records:
        content+=f"<tr><td>{'原配方' if r['recipe']=='holdout' else '温和增强'}</td><td>{r['heldout_serve']}</td><td>{r['confidence']}</td><td>{r['tp']} / {r['fp']} / {r['fn']}</td><td>{r['precision']:.1%}</td><td>{r['recall']:.1%}</td><td>{r['matched_center_error_median_px']:.2f} px</td></tr>"
    content+='</table><p>匹配框中心误差以标注 bbox 中心为参考，不是物理标记点定位真值。</p>'
    for recipe in ['holdout','augmented_holdout']:
        for camera in ['F3E','F59']:
            for held in ['01','06']:
                stem=f'{recipe}/{camera}/holdout_serve{held}_{camera}'
                content+=f'<details><summary>{recipe} / {camera} / 留出 {held} 难例（绿：标注，橙：预测）</summary><img loading="lazy" src="{stem}/worst_frames_gt_green_pred_orange.jpg"><p><a href="{stem}/diagnostic.json">数值与模型哈希</a> · <a href="{stem}/split.json">分组清单</a></p></details>'
    content+='</section><section><h2>追踪、平滑与球速检查</h2><p>1,781 个观测、186 段，均满足连续至少三帧；其中 27 段短于 2 ms，不能仅凭三帧就认定可以稳定求转速。当前对照使用同一份 v10 鲁棒平滑结果，不重复平滑、不补预测点。</p><img src="plots/track_duration.png"><p><a href="tracking_audit.json">追踪审计</a> · <a href="tracking_segments.csv">逐段长度</a> · <a href="ball_motion_audit.json">球速审计</a>。条件横向速度范围约 1.54–6.98 m/s，中位约 3.49 m/s；未钳制数值，尚非实测三维速度。</p></section><section><h2>转速与方向：同窗、同观测对照</h2><p>标记簇均衡在合成坏轨迹测试中有效，但本批真实数据没有一致收益，默认保留原权重。共同可用的 50 个整簇移除对照中，留出簇预测误差中位数为 1.81 px → 1.95 px。视频两列使用完全相同的图像和轨迹，只改变拟合权重；候选变低不代表更准确。</p><img src="plots/rotation_comparison.png"><p><a href="rotation/matched_windows.csv">逐窗口对照</a> · <a href="rotation/leave_cluster_out.csv">整簇移除敏感性</a></p>'
    for r in videos:
        content+=f'<h3>{r["serve"]}</h3><video controls preload="none" poster="posters/{r["serve"]}.jpg" src="{r["video"]}"></video><p><button onclick="step(this,-1)">前一帧</button><button onclick="step(this,1)">后一帧</button>25 FPS 慢放，源时间步长 500 μs。黑侧表示此时刻没有导出裁剪；不能推断 RAW 视野里没有球。</p>'
    content+='</section><section><h2>人工审核队列</h2><p>下列是自动筛出的待审核图，不是自动标签。四列为未过滤、当前过滤、原极性与较弱过滤。修改状态后可导出 JSON；不会写回训练集，也不会将未标注样本当作负样本。</p><button id="export">导出审核 JSON</button>'
    for q in queue:
        content+=f'<details><summary>{html.escape(q["id"])} | {q["timestamp_us"]} μs | {html.escape("、".join(q["reasons"]))}</summary><img loading="lazy" src="{q["image"]}"><p><select data-review="{q["id"]}"><option value="unreviewed">尚未审核</option><option value="needs_annotation">有标记，需要补标</option><option value="confirmed_empty">确认该帧无标记</option><option value="uncertain">不确定，不能作负样本</option><option value="geometry_problem">球心或半径需要修正</option></select></p></details>'
    content+='</section><script>const proposals='+json.dumps(queue,ensure_ascii=False)+''';
    function step(b,d){let p=b.closest('p'),v=p.previousElementSibling;v.pause();v.currentTime=Math.max(0,Math.min(v.duration,v.currentTime+d/25));}
    for(const s of document.querySelectorAll('[data-review]')){try{s.value=localStorage.getItem('pp-v11:'+s.dataset.review)||'unreviewed'}catch(e){}s.onchange=()=>{try{localStorage.setItem('pp-v11:'+s.dataset.review,s.value)}catch(e){}}}
    document.getElementById('export').onclick=()=>{const values=new Map([...document.querySelectorAll('[data-review]')].map(s=>[s.dataset.review,s.value]));const result=proposals.map(p=>({...p,review_status:values.get(p.id)}));let u=URL.createObjectURL(new Blob([JSON.stringify({scope:'human review proposals; boxes still require annotation',items:result},null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='pingpong_v11_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
    </script></html>'''
    (output/'review.html').write_text(content)


def package_review(output):
    bundle=output/'iteration_review_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path==bundle or path.suffix in {'.pt','.cache'} or 'dataset' in path.relative_to(output).parts:continue
            if path.name=='review.html':
                z.writestr('review.html',path.read_text().replace('<a href="iteration_review_bundle.zip">下载本轮审核包</a>','<span>当前为已解压的审核包</span>'))
            else:z.write(path,str(path.relative_to(output)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=Path('output/system_iteration_v11'))
    p.add_argument('--package-only',action='store_true',help='Repackage completed results after adding validation evidence')
    a=p.parse_args();root=a.root.resolve();output=a.output.resolve()
    if not a.package_only:
        records=training_plots(output);videos=render_videos(root,output);page(output,records,videos)
        (output/'evidence').mkdir(exist_ok=True)
        for name,source in [('raw_sync_report.json','output/latest_sync/sync_report.json'),('smoothing_report.json','output/latest_smoothing/smoothing_report.json'),('tracking_config.json','configs/inference_tracking.json'),('enhancement_config.json','config/marker_enhancement_v1.json')]:
            shutil.copy2(root/source,output/'evidence'/name)
    assert (output/'review.html').is_file()
    package_review(output)
    print('Review and portable bundle complete',flush=True)


if __name__=='__main__':main()
