"""Time-synchronized video overlays for controlled pixel-radius comparisons."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .motion import write_csv
from .motion_review import color
from .serve_review import conclusion
from .serve_spin import analyze_serves
from .spin_direction import describe_omega
from .spin_acceptance import LABELS

PANEL = (960, 640)
FPS = 25


@lru_cache(None)
def font(size):
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


@lru_cache(maxsize=512)
def text_image(text, face, fill):
    box=face.getbbox(text)
    image=Image.new("RGBA",(max(1,box[2]-box[0]),max(1,box[3]-box[1])))
    ImageDraw.Draw(image).text((-box[0],-box[1]),text,font=face,fill=fill)
    return image,box[:2]


class CachedDraw:
    """Reuse repeated numeric/CJK labels without caching complete video frames."""
    def __init__(self, image):
        self.image=image
        self.draw=ImageDraw.Draw(image)

    def __getattr__(self, name):
        return getattr(self.draw,name)

    def text(self, xy, text, font, fill):
        patch,offset=text_image(str(text),font,fill)
        self.image.paste(patch,(int(xy[0]+offset[0]),int(xy[1]+offset[1])),patch)


def current_fit(serve, timestamp_us, source, window_id=None):
    """No last-value hold: a fit must contain a real observation at this time."""
    candidates = []
    for window in serve["windows"]:
        if window_id is not None and window["window_id"] != window_id:
            continue
        fit = window if source == "JOINT" else window["mono"][source]
        if (fit.get("numerically_usable") and
                timestamp_us in fit.get("observation_timestamps_us", ())):
            candidates.append((window, fit))
    return min(candidates, key=lambda p: (abs((p[0]["start_us"] + p[0]["end_us"]) / 2 - timestamp_us),
                                           p[0]["start_us"]), default=None)


def baseline_source(serve, timestamp_us):
    if current_fit(serve, timestamp_us, "JOINT"):
        return "JOINT"
    preferred = conclusion(serve).get("camera")
    if preferred and current_fit(serve, timestamp_us, preferred):
        return preferred
    available = [(c, current_fit(serve, timestamp_us, c)) for c in ("F3E", "F59")]
    available = [(c, p) for c, p in available if p]
    return max(available, key=lambda p: (p[1][1].get("inlier_fraction", 0),
                                        p[1][1].get("unique_observations", 0)))[0] if available else None


def frame_value(serve, timestamp_us, source, window_id=None):
    selected = current_fit(serve, timestamp_us, source, window_id) if source else None
    result = {"timestamp_us": int(timestamp_us), "source": source, "available": selected is not None,
              "rps": None, "rpm": None, "axis_camera": None, "axis": None,
              "window_start_us": None, "window_end_us": None, "window_id": None, "comparison_window_id": window_id,
              "directions": {}, "independent": {}, "state": "当前时刻无可靠观测"}
    for camera in ("F3E", "F59"):
        mono = current_fit(serve, timestamp_us, camera, window_id)
        result["independent"][camera] = {"rps": mono[1]["candidate_rps"], "rpm": mono[1]["candidate_spin_rpm"]} if mono else None
    if selected is None:
        if source:
            result["state"] = "此半径下同类拟合不可用"
        return result
    window, fit = selected
    reference = "F3E" if source == "JOINT" else source
    omega = np.array(fit["omega_camera_rad_s"], float)
    near_zero = fit.get("rotation_signal") == "near_zero_unresolved_direction"
    magnitude = float(np.linalg.norm(omega))
    axis = (omega / magnitude).tolist() if magnitude > 1e-8 and not near_zero else None
    group=serve.get("joint_summary",{}) if source=="JOINT" else serve.get("views",{}).get(source,{}).get("aligned_summary",{})
    group_status=group.get("engineering_direction_status","not_assessed")
    state="近零候选，方向未分辨" if near_zero else ("离线窗口候选" if group.get("engineering_stable") else "单窗候选，整组尚未稳定")
    result.update(rps=fit["candidate_rps"], rpm=fit["candidate_spin_rpm"], axis_camera=reference,
                  axis=axis, window_start_us=window["start_us"], window_end_us=window["end_us"],
                  window_id=window["window_id"], state=state, group_status=group_status)
    r = np.array(serve["R_F59_to_F3E"]) if serve["R_F59_to_F3E"] is not None else None
    for camera in ("F3E", "F59"):
        local = omega if camera == reference else ((r.T @ omega if reference == "F3E" else r @ omega) if r is not None else None)
        result["directions"][camera] = describe_omega(np.zeros(3) if near_zero else local) if local is not None else None
    return result


def compute_variants(tracks, dataset, config, previous, output, factors=(1., 1.1, 1.2, 1.5)):
    factors = tuple(float(f) for f in factors)
    if not factors or factors[0] != 1 or len(factors) > 4 or len(set(factors)) != len(factors) or any(not math.isfinite(f) or f <= 0 for f in factors):
        raise ValueError("use 1 to 4 distinct positive radius factors, starting with baseline 1.0")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = analyze_serves(tracks, dataset, config, previous)
    fixed = {s["serve"]: {k: s[k] for k in ("R_F59_to_F3E", "orientation_training_serves")} for s in base["serves"]}
    variants = []
    for factor in factors:
        report = base if factor == 1 else analyze_serves(tracks, dataset, config, previous,
                    radius_multiplier=factor, fixed_relative_orientations=fixed)
        name = ("r" + format(factor * 100, ".10g")).replace(".", "p")
        folder = output / name
        folder.mkdir()
        path = folder / "serve_spin_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        variants.append({"name": name, "factor": factor, "report_path": str(path.relative_to(output))})
        print(f"Fitted radius factor {factor:g}", flush=True)
    index = {"variants": variants, "only_sphere_radius_varies": True,
             "relative_orientation_fixed_to_baseline": True, "physical_accuracy_validated": False}
    (output / "variants.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def arrow(draw, center, direction, fill):
    cx, cy = center
    draw.ellipse((cx-42, cy-42, cx+42, cy+42), outline="#697f91", width=2)
    if direction is None or direction["front_direction_dx"] is None:
        draw.text((cx-33, cy-10), "未分辨", font=font(16), fill="#bac5d0")
        return
    dx, dy = direction["front_direction_dx"], direction["front_direction_dy"]
    start, end = (cx-23*dx, cy-23*dy), (cx+32*dx, cy+32*dy)
    draw.line([start, end], fill=fill, width=5)
    base = (end[0]-13*dx, end[1]-13*dy)
    draw.polygon([end, (base[0]-7*dy, base[1]+7*dx), (base[0]+7*dy, base[1]-7*dx)], fill=fill)


def render_panel(images, frame_rows, marker_rows, marker_runs, value, serve, radius_px, factor, first_us):
    canvas = Image.new("RGB", PANEL, "#111c28")
    draw = CachedDraw(canvas)
    fill = "#89e2b5" if value["source"] == "JOINT" else "#f7ce83"
    t = value["timestamp_us"]
    draw.text((18, 10), f"假设球半径 {radius_px:.1f} px  ×{factor:g}  |  +{(t-first_us)/1000:.1f} ms", font=font(22), fill="#e7eef5")
    source_label = {"JOINT": "双视角联合", "F3E": "F3E 单侧", "F59": "F59 单侧", None: "无可用模型"}[value["source"]]
    headline = f"{source_label}  {value['rps']:.2f} 转/秒  |  {value['rpm']:.0f} 转/分" if value["available"] else f"{source_label}  N/A — {value['state']}"
    draw.text((18, 48), headline, font=font(30), fill=fill if value["available"] else "#c5cdd5")
    axis = "方向未分辨" if value["axis"] is None else "(" + ", ".join(f"{x:+.3f}" for x in value["axis"]) + ")"
    draw.text((18, 96), f"有向单位轴 [{value['axis_camera'] or '—'}]  {axis}", font=font(18), fill="#c4d2df")
    draw.text((18, 123), "方向为相机局部坐标；箭头显示可见前表面中心的运动", font=font(16), fill="#92a8bb")
    for j, camera in enumerate(("F3E", "F59")):
        x, y = 20+j*480, 178
        mono = value["independent"][camera]
        label = f"{camera} 独立：{mono['rps']:.2f} 转/秒" if mono else f"{camera} 独立：N/A"
        draw.text((x, 149), label, font=font(18), fill="#dae5ed")
        if camera in images:
            patch = images[camera].copy()
            pd = CachedDraw(patch)
            rr = radius_px * 1.25
            pd.ellipse((160-rr,160-rr,160+rr,160+rr), outline="#f7c666", width=2)
            pd.line((154,160,166,160), fill="#f7c666", width=1)
            pd.line((160,154,160,166), fill="#f7c666", width=1)
            for marker in marker_rows[camera]:
                col = color(marker["track_id"])
                run = marker_runs[camera][marker["segment_id"]]
                tail = [(float(p["center_x_px"])*1.25,float(p["center_y_px"])*1.25) for p in run if 0<=t-int(p["timestamp_us"])<=4000]
                if len(tail)>1:
                    pd.line(tail, fill=col, width=2)
                px, py = float(marker["center_x_px"])*1.25, float(marker["center_y_px"])*1.25
                pd.ellipse((px-4,py-4,px+4,py+4), outline=col, width=2)
                pd.text((px+5,py-9), str(marker["track_id"]), font=font(12), fill=col)
            canvas.paste(patch, (x,y))
            row = frame_rows[camera]
            draw.text((x, 505), f"帧 {row['frame_index']} | 已观测点 {len(marker_rows[camera])}", font=font(16), fill="#becfdd")
        else:
            draw.rectangle((x,y,x+320,y+320), fill="#172331")
            draw.text((x+17,y+140), "未导出此时刻球裁剪", font=font(21), fill="#9caabb")
        d = value["directions"].get(camera)
        arrow(draw, (x+394, 326), d, fill)
        draw.text((x+350, 385), LABELS[d["front_motion_label"]] if d else "未确定", font=font(17), fill=fill)
        sense = d["image_z_spin_sense"] if d else "unresolved"
        spin = {"clockwise":"视轴分量：顺时针", "counterclockwise":"视轴分量：逆时针",
                "small_view_axis_component":"视轴分量较小", "unresolved":"视轴方向未分辨"}[sense]
        draw.text((x, 533), spin, font=font(17), fill="#bdcddd")
        if d and d["axis_x"] is not None:
            draw.text((x, 561), "局部轴 " + ", ".join(f"{d[k]:+.2f}" for k in ("axis_x","axis_y","axis_z")), font=font(16), fill="#a9c4d5")
    if value["available"]:
        text=f"拟合窗 +{(value['window_start_us']-first_us)/1000:.1f}–{(value['window_end_us']-first_us)/1000:.1f} ms | {value['state']}"
    else:
        text="缺测与拒绝不填 0；未沿用上一窗口的转速"
    draw.text((18, 596), text, font=font(17), fill="#c8d4df")
    draw.text((18, 621), f"源时间 {t} μs | 离线估计随播放更新 | 半径为裁剪像素尺度", font=font(13), fill="#8ca1b3")
    return canvas


def matched_summary(variants, serves):
    rows=[]
    for serve_name in serves:
        base=serves[serve_name][0]
        choice=conclusion(base)
        source="JOINT" if choice["mode"]=="experimental_two_view" else choice.get("camera")
        if source is None:
            rows.append({"serve":serve_name,"source":None,"matched_windows":0,"rps_by_variant":{v['name']:None for v in variants}})
            continue
        shared=[]
        for windows in zip(*[s["windows"] for s in serves[serve_name]]):
            fits=[w if source=="JOINT" else w["mono"][source] for w in windows]
            if all(f["numerically_usable"] for f in fits):
                shared.append([f["candidate_rps"] for f in fits])
        rows.append({"serve":serve_name,"source":source,"matched_windows":len(shared),
                     "rps_by_variant":{v['name']:float(np.median(np.array(shared)[:,i])) if shared else None for i,v in enumerate(variants)}})
    return rows


def build_radius_media(root, tracks_path, videos=True):
    root=Path(root)
    index=json.loads((root/"variants.json").read_text())
    variants=index["variants"]
    reports=[json.loads((root/v["report_path"]).read_text()) for v in variants]
    serves={s["serve"]:[next(t for t in r["serves"] if t["serve"]==s["serve"]) for r in reports] for s in reports[0]["serves"]}
    for folder in ("videos","posters","previews","plots"):
        (root/folder).mkdir(exist_ok=True)
    from .marker_spin import read_tracks
    from .trajectory_smoothing import csv_rows as effective_csv_rows
    effective,_,_=read_tracks(tracks_path,reports[0]['dataset'],
                               smoothing_config=reports[0]['config'].get('trajectory_smoothing'))
    tracks=effective_csv_rows(effective)
    csv_rows=[]
    media=[]
    for name, cases in serves.items():
        frame_maps,folders,by_time,runs={},{},{},{}
        for camera in ("F3E","F59"):
            folder=Path(reports[0]["dataset"])/"unlabeled_inference"/camera/f"{name}_{camera}"
            folders[camera]=folder
            with (folder/"manifest.csv").open(encoding="utf-8-sig") as stream:
                frame_maps[camera]={int(row["timestamp_us"]):row for row in csv.DictReader(stream)}
            by_time[camera],runs[camera]=defaultdict(list),defaultdict(list)
            for row in tracks:
                if row["group"]==f"{name}_{camera}":
                    by_time[camera][int(row["timestamp_us"])].append(row)
                    runs[camera][row["segment_id"]].append(row)
            for run in runs[camera].values(): run.sort(key=lambda p:int(p["timestamp_us"]))
        times=sorted(set(frame_maps["F3E"])|set(frame_maps["F59"]))
        radii=[reports[i]["sources"][f"F3E/{name}_F3E"]["sphere_radius_px"] for i in range(len(variants))]
        frame_values=[]
        for t in times:
            source=baseline_source(cases[0],t)
            selected=current_fit(cases[0],t,source) if source else None
            pinned_id=selected[0]["window_id"] if selected else None
            values=[frame_value(case,t,source,pinned_id) for case in cases]
            frame_values.append(values)
            for variant,radius,value in zip(variants,radii,values):
                row={"serve":name,"timestamp_us":t,"variant":variant["name"],"radius_factor":variant["factor"],"radius_px":radius,
                     "source_locked_to_baseline":source,"available":value["available"],"candidate_rps":value["rps"],"candidate_rpm":value["rpm"],
                     "axis_camera":value["axis_camera"],"window_id":value["window_id"],"comparison_window_id":value["comparison_window_id"],"window_start_us":value["window_start_us"],"window_end_us":value["window_end_us"],"state":value["state"]}
                for j,key in enumerate(("axis_x","axis_y","axis_z")): row[key]=value["axis"][j] if value["axis"] else None
                for camera in ("F3E","F59"):
                    d=value["directions"].get(camera)
                    row[camera+"_front_motion"]=d["front_motion_label"] if d else None
                    row[camera+"_independent_rps"]=value["independent"][camera]["rps"] if value["independent"][camera] else None
                csv_rows.append(row)
        representative=max(range(len(times)),key=lambda i:(sum(v["available"] for v in frame_values[i]),
            min(len(by_time[c].get(times[i],[])) for c in frame_maps),-abs(i-len(times)/2)))
        def render(i):
            t=times[i]
            images={c:Image.open(folders[c]/frame_maps[c][t]["image"]).convert("RGB").resize((320,320)) for c in frame_maps if t in frame_maps[c]}
            rows={c:frame_maps[c].get(t) for c in frame_maps}
            markers={c:by_time[c].get(t,[]) for c in frame_maps}
            canvas=Image.new("RGB",(PANEL[0]*2,PANEL[1]*2),"#101923")
            for j,(variant,case,value,radius) in enumerate(zip(variants,cases,frame_values[i],radii)):
                panel=render_panel(images,rows,markers,runs,value,case,radius,variant["factor"],times[0])
                canvas.paste(panel,((j%2)*PANEL[0],(j//2)*PANEL[1]))
            return canvas
        render(0).save(root/"posters"/f"{name}.png")
        render(representative).save(root/"previews"/f"{name}.png")
        if videos:
            destination=root/"videos"/f"{name}_radius_compare.mp4"
            with (destination.with_suffix(".log")).open("w") as log:
                process=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1280","-r",str(FPS),"-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","23","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)],stdin=subprocess.PIPE,stderr=log)
                try:
                    for i in range(len(times)): process.stdin.write(np.asarray(render(i)).tobytes())
                finally:
                    process.stdin.close()
                if process.wait()!=0: raise RuntimeError(f"video failed: {destination}")
        item={"serve":name,"frames":len(times),"timestamps_us":times,"values":frame_values,"radii_px":radii,
              "video":f"videos/{name}_radius_compare.mp4" if videos else None,"poster":f"posters/{name}.png",
              "representative_frame":representative,"fps":FPS,"group_baseline_conclusion":conclusion(cases[0])}
        media.append(item)
        fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
        for j,v in enumerate(variants):
            ax.scatter([(t-times[0])/1000 for t,values in zip(times,frame_values) if values[j]["available"]],
                       [values[j]["rps"] for values in frame_values if values[j]["available"]],s=7,label=f"R={radii[j]:.1f} px ({v['factor']:g}x)",alpha=.65)
        ax.set(xlabel="Source time since first frame (ms)",ylabel="Displayed candidate spin (rps)",ylim=(0,None),title=name+" | comparison source locked to baseline at each timestamp")
        ax.grid(alpha=.2);ax.legend();fig.savefig(root/"plots"/f"{name}.png",dpi=140);plt.close(fig)
        print(f"Rendered {name}: {len(times)} frames, {len(variants)} radii",flush=True)
    summary=matched_summary(variants,serves)
    write_csv(root/"frame_values.csv",csv_rows)
    write_csv(root/"radius_summary.csv",[{"serve":s["serve"],"source":s["source"],"matched_windows":s["matched_windows"],**s["rps_by_variant"]} for s in summary])
    result={"variants":variants,"serves":media,"matched_window_summary":summary,"physical_accuracy_validated":False,
            "comparison": "same pixels, tracks, timestamps, fixed camera orientation; only model pixel radius varies",
            "display": "offline window estimate at real observed timestamps; no last-value hold; source locked to baseline across radii",
            "directions": "signed camera-frame axis and front-surface motion; not world top/back/side labels"}
    (root/"frame_overlays.json").write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    write_html(result,root)
    return result


def write_html(result, root):
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PingPong 半径、转速与方向同步对比</title><style>body{font:16px system-ui;max-width:1450px;margin:24px auto;padding:0 18px;background:#f3f5f8;color:#192332;line-height:1.6}canvas{display:block;width:100%;background:#101923}button,select{font-size:16px;padding:6px;margin:5px}input[type=range]{width:60%}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.note{padding:14px;background:#fff0ca}video{display:none}img{max-width:100%}a{color:#185b99}</style>
<h1>半径 × 转速 × 方向：随视频时间同步查看</h1><p class="note">这是离线窗口候选值，随播放更新，不是在线相机测量。四组采用同一原图、轨迹、时间窗口与相机相对方向，只改变球面模型的像素半径。主数值的数据来源也按基准组锁定，便于同类比较；某组拟合失败显示 N/A，不复用旧值、不填 0。人工比较应同时看黄色球轮廓与原图边缘，不能仅因转速较低就选定半径。</p>
<p>四组从左上到右下排列。黄色圆为假设球面边界；箭头为相机前表面中心的运动，有向轴使用相机局部坐标。大于 128 px 的半径超出裁剪半宽，属于尺度敏感性试验。25 FPS 按当前 500 μs 源采样约慢放 80 倍。</p>
<p><a href="frame_values.csv">逐帧显示值 CSV</a> · <a href="radius_summary.csv">相同窗口半径对照 CSV</a> · <a href="frame_overlays.json">完整显示数据</a></p>
<label>发球 <select id="serve"></select></label><label>放大查看 <select id="focus"><option value="all">四组同屏</option></select></label>
<canvas id="canvas" width="1920" height="1280"></canvas><video id="video" preload="metadata" playsinline></video>
<div><button id="play">播放 / 暂停</button><button id="prev">上一帧</button><button id="next">下一帧</button><input id="seek" type="range" min="0" step="1" value="0"><span id="position"></span></div><p id="message"></p><p><a id="download">打开当前发球 MP4</a> · <a id="plot">转速曲线图</a></p>
<table><thead><tr><th>半径 px / 倍率</th><th>来源</th><th>转/秒</th><th>转/分</th><th>有向单位轴 / 参考相机</th><th>F3E 前表面</th><th>F59 前表面</th><th>状态</th></tr></thead><tbody id="numbers"></tbody></table>
<h2>相同有效窗口的转速中位数</h2><p>仅对四组都通过数值检查的同一批窗口比较；这不是实测精度，也不代表不转发球的下限。第 05 次缺乏整组稳定候选，仍保持未确定。</p><div id="summary"></div>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);const video=$('video'),canvas=$('canvas'),ctx=canvas.getContext('2d');let current=0,poster=new Image(),decodedIndex=null;
const labels={up:'↑ 上',down:'↓ 下',left:'← 左',right:'→ 右',up_left:'↖ 左上',up_right:'↗ 右上',down_left:'↙ 左下',down_right:'↘ 右下',unresolved:'未确定'};
DATA.serves.forEach((s,i)=>$('serve').add(new Option(s.serve,i)));DATA.variants.forEach((v,i)=>$('focus').add(new Option('半径倍率 '+v.factor,i)));
function draw(){const s=DATA.serves[current],i=decodedIndex===null?Math.min(s.frames-1,Math.max(0,Math.floor(video.currentTime*s.fps+1e-4))):Math.min(s.frames-1,decodedIndex);$('seek').value=i;$('position').textContent=`${i+1}/${s.frames} 帧 · 源时间 ${s.timestamps_us[i]} μs`;const focus=$('focus').value;const width=focus==='all'?1920:960,height=focus==='all'?1280:640;if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;const source=video.readyState>=2?video:poster;if(source===video||poster.complete){if(focus==='all')ctx.drawImage(source,0,0,1920,1280);else{const k=Number(focus);ctx.drawImage(source,(k%2)*960,Math.floor(k/2)*640,960,640,0,0,960,640)}}$('numbers').innerHTML=s.values[i].map((v,j)=>`<tr><td>${s.radii_px[j].toFixed(1)} / ${DATA.variants[j].factor}</td><td>${v.source||'无'}</td><td>${v.rps==null?'N/A':v.rps.toFixed(2)}</td><td>${v.rpm==null?'N/A':v.rpm.toFixed(0)}</td><td>${v.axis?v.axis.map(x=>x.toFixed(3)).join(', ')+' / '+v.axis_camera:'未分辨'}</td><td>${labels[v.directions.F3E?.front_motion_label]||'未确定'}</td><td>${labels[v.directions.F59?.front_motion_label]||'未确定'}</td><td>${v.state}</td></tr>`).join('');}
function load(){video.pause();decodedIndex=null;const s=DATA.serves[current];video.src=s.video||'';video.load();poster=new Image();poster.onload=draw;poster.src=s.poster;$('seek').max=s.frames-1;$('seek').value=0;$('download').href=s.video||s.poster;$('plot').href='plots/'+s.serve+'.png';$('message').textContent='数值使用源时间，不使用播放 FPS。可暂停或逐帧检查；方向尚非世界坐标中的上旋/下旋。';draw();}
$('serve').onchange=()=>{current=Number($('serve').value);load()};$('focus').onchange=draw;$('play').onclick=()=>{if(video.paused)video.play().catch(e=>$('message').textContent=e.message);else video.pause()};function seek(i){video.pause();video.currentTime=Math.max(0,Math.min(DATA.serves[current].frames-1,i))/DATA.serves[current].fps;}
$('seek').oninput=()=>seek(Number($('seek').value));$('prev').onclick=()=>seek(Number($('seek').value)-1);$('next').onclick=()=>seek(Number($('seek').value)+1);video.onseeked=draw;video.ontimeupdate=draw;video.onloadeddata=draw;video.onerror=()=>{$('message').textContent='视频加载失败，可使用下面的 MP4 链接直接打开。'};if(video.requestVideoFrameCallback){const presented=(now,metadata)=>{decodedIndex=Math.max(0,Math.round(metadata.mediaTime*DATA.serves[current].fps));draw();video.requestVideoFrameCallback(presented)};video.requestVideoFrameCallback(presented)}else{function tick(){if(!video.paused)draw();requestAnimationFrame(tick)}requestAnimationFrame(tick)};
$('summary').innerHTML='<table><tr><th>发球</th><th>数据来源</th><th>共同窗口数</th>'+DATA.variants.map(v=>'<th>×'+v.factor+' RPS</th>').join('')+'</tr>'+DATA.matched_window_summary.map(s=>'<tr><td>'+s.serve+'</td><td>'+(s.source||'未确定')+'</td><td>'+s.matched_windows+'</td>'+DATA.variants.map(v=>'<td>'+(s.rps_by_variant[v.name]==null?'N/A':s.rps_by_variant[v.name].toFixed(2))+'</td>').join('')+'</tr>').join('')+'</table>';load();</script></html>'''
    page=page.replace("__DATA__",json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/"))
    (root/"review.html").write_text(page,encoding="utf-8")
