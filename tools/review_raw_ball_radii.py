"""Compare sphere-radius hypotheses directly against actual RAW ball boundaries."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pingpong_event_pipeline.raw_sync import IndexedEvents, load_serves
from pingpong_event_pipeline.radius_review import CachedDraw,font
from pingpong_event_pipeline.motion import write_csv
from render_ball_raw_review import gray_frame


def build(dataset,output,cache,radius):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    overlays={s['serve']:s for s in json.loads((Path(radius)/'frame_overlays.json').read_text())['serves']}
    records=[];names=[]
    for s in load_serves(dataset):
        name=s['serve'];o=overlays[name]
        maps={c:{int(r['timestamp_us']):r for r in s['cameras'][c]['rows']} for c in ['F3E','F59']}
        common=set(maps['F3E'])&set(maps['F59'])
        candidates=[i for i,t in enumerate(o['timestamps_us']) if t in common]
        i=max(candidates,key=lambda k:(o['values'][k][0]['available'] and o['values'][k][0]['source']=='JOINT',
                                      sum(v['available'] for v in o['values'][k]),-abs(k-len(o['timestamps_us'])/2)))
        t=o['timestamps_us'][i];canvas=Image.new('RGB',(1440,940),'#142130');draw=CachedDraw(canvas)
        draw.text((12,8),f'{name} | RAW 同一事件窗 [{t-1000}, {t}) μs | 直接对照球体边界',font=font(23),fill='white')
        draw.text((12,46),'每行图像、球心、事件时间完全相同；圈仅表示不同半径假设。数值仍为离线转速候选。',font=font(19),fill='#ebd28e')
        for j,c in enumerate(['F3E','F59']):
            row=maps[c][t];r=float(row['radius_px']);cx=float(row['center_x_px']);cy=float(row['center_y_px'])
            source=IndexedEvents(np.load(Path(cache)/f'{name}_{c}.npy',mmap_mode='r'))
            image=gray_frame(source.window(t-1000,t))
            crop_side=max(8,round(4*r))
            crop=cv2.getRectSubPix(image,(crop_side,crop_side),(cx,cy))
            crop=cv2.resize(crop,(320,320),interpolation=cv2.INTER_NEAREST)
            scale=320/crop_side
            for k,factor in enumerate([1,1.1,1.2,1.5]):
                x=20+k*360;y=133+j*400;canvas.paste(Image.fromarray(crop).convert('RGB'),(x,y))
                rr=r*factor*scale
                draw.ellipse((x+160-rr,y+160-rr,x+160+rr,y+160+rr),outline='#ffbd53',width=2)
                draw.text((x,y-29),f'{c} | R={r*factor:.1f} RAW px ×{factor:g}',font=font(17),fill='#95e5c3')
                value=o['values'][i][k];rate=f"{value['rps']:.2f} 转/秒" if value['available'] else 'N/A'
                draw.text((x,y+322),f'{rate} | {value["source"] or "无有效拟合"}',font=font(18),fill='white')
                records.append({'serve':name,'camera':c,'timestamp_us':t,'source_frame':row['frame_index'],
                                'sphere_radius_multiplier':factor,'hypothesized_raw_radius_px':r*factor,
                                'candidate_rps':value['rps'],'source':value['source'],
                                'radius_independently_measured':False})
        draw.text((12,910),'若假设圆明显偏离 RAW 球边界，不能仅因转速数值更低就采用；原球心/轮廓本身也需审核。',font=font(18),fill='#ebd28e')
        filename=f'{name}_raw_radius.png';canvas.save(output/filename);names.append((name,filename))
    write_csv(output/'raw_radius_examples.csv',records)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RAW 球半径与转速对照</title><style>body{font:16px system-ui;max-width:1440px;margin:25px auto;line-height:1.7}img{max-width:100%}</style><h1>直接在 RAW 球轮廓上比较半径假设</h1><p>每次发球两行取完全相同的绝对时间，列间保持原像素与球心不变。选择有双侧裁剪且尽量有联合候选的时刻，没有根据数值高低挑帧。图中黄圈是假设球半径，并非新的边界测量。转速候选沿用同一时间、同一拟合窗口的四种球面半径实验。</p><p><a href="../review.html">返回视频审核</a> · <a href="raw_radius_examples.csv">对应数值</a></p>'
    for name,file in names:page+=f'<h2>{name}</h2><img src="{file}">'
    (output/'review.html').write_text(page+'</html>',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--radius',type=Path,default=Path('output/latest_radius'))
    a=p.parse_args();build(a.dataset,a.output,a.cache,a.radius)
