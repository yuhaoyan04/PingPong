"""Plot texture repeat periods and side-by-side repeated appearances."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pingpong_event_pipeline.appearance_period import texture_feature,recurrence,common_peaks
from pingpong_event_pipeline.raw_sync import load_serves
from pingpong_event_pipeline.radius_review import CachedDraw,font
from pingpong_event_pipeline.serve_review import conclusion
from pingpong_event_pipeline.motion import write_csv


def build(dataset,output,baseline):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    base={s['serve']:s for s in json.loads(Path(baseline).read_text())['serves']}
    reports=[];peak_rows=[];common_rows=[]
    for spec in load_serves(dataset):
        name=spec['serve']; results={}; images={};times={}
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        expected=conclusion(base[name]).get('rps')
        for j,c in enumerate(['F3E','F59']):
            source=spec['cameras'][c];rows=source['rows'];folder=Path(source['folder'])
            images[c]=[cv2.imread(str(folder/r['image']),0) for r in rows]
            times[c]=np.array([int(r['timestamp_us']) for r in rows])
            features=[texture_feature(im,float(source['metadata']['crop_radius_scale'])) for im in images[c]]
            result=recurrence(features,times[c]);results[c]=result
            for p in result['peaks']:peak_rows.append({'serve':name,'camera':c,**p})
            x=result['lags']*result['step_us']/1000;ax=axes[0,j]
            ax.plot(x,result['scores'],alpha=.4,label='Mean texture correlation')
            ax.plot(x,result['smoothed'],label='Smoothed, sigma=0.5 ms')
            for p in result['peaks']:
                if p['mean_correlation']>=.2:
                    ax.plot(p['delay_us']/1000,p['mean_correlation'],'o')
                    ax.annotate(f"{p['delay_us']/1000:g} ms",(p['delay_us']/1000,p['mean_correlation']),fontsize=8,xytext=(3,5),textcoords='offset points')
            if expected:
                ax.axvline(1000/expected,color='tab:red',linestyle='--',label=f'Existing fit 1/f = {1000/expected:.1f} ms')
            ax.set(title=name+' '+c,xlabel='Repeat delay (ms)',ylabel='Mean normalized correlation',ylim=(-.25,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
            extent=[0,(times[c][-1]-times[c][0])/1000,(times[c][-1]-times[c][0])/1000,0]
            im=axes[1,j].imshow(result['matrix'],vmin=-.25,vmax=1,cmap='viridis',extent=extent)
            axes[1,j].set(xlabel='Time since own crop start (ms)',ylabel='Time since own crop start (ms)',title='Texture similarity; separate local time origin')
            fig.colorbar(im,ax=axes[1,j])
        fig.savefig(output/f'{name}_recurrence.png',dpi=140);plt.close(fig)
        matched=common_peaks(results['F3E']['peaks'],results['F59']['peaks'])
        for m in matched:common_rows.append({'serve':name,**m})
        sheets=[]
        for n,m in enumerate(matched):
            canvas=Image.new('RGB',(1200,770),'#142130');draw=CachedDraw(canvas)
            draw.text((12,8),f"{name} | 双侧纹理重复延迟约 {m['mean_repeat_delay_us']/1000:.2f} ms；1/T={m['equivalent_1_over_delay_rps']:.2f} /秒",font=font(23),fill='white')
            draw.text((12,45),'下列是各相机自身的纹理重复示例；两行独立选取起点，不能作为双目同时帧使用。',font=font(18),fill='#ebd28e')
            for j,c in enumerate(['F3E','F59']):
                k=round(m[f'{c}_delay_us']/results[c]['step_us']);matrix=results[c]['matrix'];N=len(images[c])
                if 2*k<N:
                    scores=(np.diag(matrix,k)[:N-2*k]+np.diag(matrix,k)[k:])/2
                else:scores=np.diag(matrix,k)
                start=int(np.argmax(scores)); offsets=[0,round(k/2),k,2*k]; y=110+j*310
                for column,offset in enumerate(offsets):
                    idx=start+offset;x=12+column*300
                    if idx<N:
                        canvas.paste(Image.fromarray(images[c][idx]).convert('RGB').resize((240,240)),(x,y))
                        label=f'{c} t={times[c][idx]} μs'
                    else:label=f'{c} 超出已有裁剪'
                    draw.text((x,y-27),f"{label}",font=font(15),fill='#95e5c3')
                    draw.text((x,y+242),f"+{offset*results[c]['step_us']/1000:g} ms",font=font(17),fill='white')
            draw.text((12,743),'重复纹理可能对应一周、多周或相似标记的分数周，不能直接认定物理转速。',font=font(17),fill='#ebd28e')
            filename=f'{name}_repeat_{n+1}.png';canvas.save(output/filename);sheets.append(filename)
        report={'serve':name,'existing_baseline_candidate_rps':expected,
                'views':{c:{'frames':len(times[c]),'step_us':r['step_us'],'peaks':r['peaks']} for c,r in results.items()},
                'common_repeat_branches':matched,'repeat_sheets':sheets,'physical_spin_accepted':False}
        reports.append(report);print(name,matched,flush=True)
    write_csv(output/'appearance_peaks.csv',peak_rows);write_csv(output/'common_repeats.csv',common_rows)
    report={'method':'interior texture correlation; no YOLO IDs or SO(3) radius parameter; existing centre/scale-normalized crops retained',
            'minimum_pairs':16,'minimum_lag_frames':6,'maximum_lag_frames':160,'smoothing_sigma_frames':1,
            'peak_prominence':.04,'common_peak_minimum_correlation':.2,'common_delay_tolerance':'max(1000 us, 8% mean delay)',
            'physical_spin_accepted':False,'serves':reports,
            'limitations':['Not an independent measurement of camera synchronization or physical clock scale.',
                           'Texture symmetries, partial/multiple revolutions, moving noise and normalization can create false repeat branches.',
                           'No peaks is insufficient evidence, not zero spin. No fitted rate is changed using this diagnostic.']}
    (output/'appearance_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>标记纹理重复周期审核</title><style>body{font:16px system-ui;max-width:1200px;margin:25px auto;line-height:1.7}img{max-width:100%}section{margin-top:30px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ddd}.note{background:#fff0cd;padding:15px}</style><h1>标记纹理重复周期：转速候选的交叉检查</h1><p>直接比较球体内部纹理，不使用 YOLO 标记簇 ID，也不使用球面 SO(3) 拟合的半径参数；仍依赖现有球心与尺度归一化裁剪。只计算相机自身的连续图像，未对两路作强制时移。</p><p class="note">1/T 是纹理重复频率，尚不能等同于物理转速。一次纹理重复可能是一周、多周或相似标记的分数周。需要结合独特标记身份及几何验证；没有明显重复也不能当作 0 转。图中红线只对照既有候选，不用于选择纹理峰。</p><p><a href="../review.html">返回 RAW、轨迹与转速视频</a> · <a href="appearance_report.json">完整诊断</a> · <a href="common_repeats.csv">双侧重复分支</a></p><table><tr><th>发球</th><th>双侧相近的纹理重复延迟</th><th>相应 1/T（次/秒）</th></tr>'''
    for r in reports:
        ms='、'.join(f"{m['mean_repeat_delay_us']/1000:.2f} ms" for m in r['common_repeat_branches']) or '未找到可信的共同分支'
        hz='、'.join(f"{m['equivalent_1_over_delay_rps']:.2f}" for m in r['common_repeat_branches']) or 'N/A'
        page+=f"<tr><td>{r['serve']}</td><td>{ms}</td><td>{hz}</td></tr>"
    page+='</table>'
    for r in reports:
        page+=f"<section><h2>{r['serve']}</h2><img src=\"{r['serve']}_recurrence.png\">"
        for file in r['repeat_sheets']:page+=f'<img src="{file}">'
        page+='</section>'
    (output/'review.html').write_text(page+'</html>',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('output/latest_radius/r100/serve_spin_report.json'))
    a=p.parse_args();build(a.dataset,a.output,a.baseline)
