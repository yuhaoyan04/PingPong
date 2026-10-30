"""Standalone motion/spin plots, annotated videos and a local interactive page."""
from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .marker_spin import read_tracks, sha256
from .motion import ball_motion, stereo_motion, write_csv


def color(track_id):
    rgba = plt.get_cmap("tab20")(int(track_id) % 20)
    return tuple(int(x * 255) for x in rgba[:3])


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def attach_spin(rows, windows):
    for row in rows:
        candidates = [w for w in windows if w["numerically_usable"] and w["start_us"] <= row["timestamp_us"] < w["end_us"]]
        chosen = min(candidates, key=lambda w: abs((w["start_us"] + w["end_us"]) / 2 - row["timestamp_us"]), default=None)
        row["window_candidate_rpm"] = chosen["candidate_spin_rpm"] if chosen else None
        row["spin_window_id"] = chosen["window_id"] if chosen else None


def group_plot(rows, markers, windows, path, crop_radius=128 / 1.2):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    valid = np.array([r["track_support"] == "ring_supported" for r in rows])
    centers = np.array([[r["center_x_px"], r["center_y_px"]] for r in rows])
    t = np.array([r["seconds_since_group_start"] * 1000 for r in rows])
    ax = axes[0, 0]
    ax.scatter(*centers[~valid].T, color=".75", s=8, label="fallback (excluded from velocity)")
    points = ax.scatter(*centers[valid].T, c=t[valid], cmap="viridis", s=12)
    fig.colorbar(points, ax=ax, label="Physical time since group start (ms)")
    ax.set(title="Ball centre in original sensor coordinates", xlabel="Sensor x (px)", ylabel="Sensor y (px)")
    ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim")
    ax = axes[0, 1]
    ax.add_patch(plt.Circle((128, 128), crop_radius, fill=False, color=".6"))
    segments = defaultdict(list)
    for m in markers:
        segments[m["segment_id"]].append(m)
    for run in segments.values():
        ax.plot([m["center_x_px"] for m in run], [m["center_y_px"] for m in run], ".-", markersize=2,
                color=np.array(color(run[0]["track_id"])) / 255, linewidth=1)
    ax.set(title="Marker trajectories in stabilized crop (>=3 frames)", xlim=(0, 256), ylim=(256, 0), xlabel="Crop x (px)", ylabel="Crop y (px)")
    ax.set_aspect("equal")
    ax = axes[1, 0]
    speed = np.array([r["image_speed_px_s"] if r["image_speed_px_s"] is not None else np.nan for r in rows])
    projected = np.array([r["candidate_transverse_speed_mps"] if r["candidate_transverse_speed_mps"] is not None else np.nan for r in rows])
    breaks = np.flatnonzero((np.diff([r["timestamp_us"] for r in rows]) > 750)
                           | (np.diff([r["frame_index"] for r in rows]) != 1)) + 1
    runs = np.split(np.arange(len(rows)), breaks)
    for run in runs:
        ax.plot(t[run], speed[run], color="tab:blue")
    ax.set(xlabel="Physical time (ms)", ylabel="Image speed (px/s)", title="Ball translation; gaps are not interpolated")
    twin = ax.twinx()
    for run in runs:
        twin.plot(t[run], projected[run], "--", color="tab:orange")
    twin.set_ylabel("Diameter-scaled transverse estimate (m/s)", color="tab:orange")
    ax = axes[1, 1]
    good = [w for w in windows if w["numerically_usable"]]
    start = rows[0]["timestamp_us"]
    ax.scatter([(w["start_us"] - start) / 1000 for w in good], [w["candidate_spin_rpm"] for w in good], color="tab:orange", label="Accepted numerical windows")
    rejected = [w for w in windows if not w["numerically_usable"] and "raw_fit_rpm" in w]
    ax.scatter([(w["start_us"] - start) / 1000 for w in rejected], [w["raw_fit_rpm"] for w in rejected], marker="x", color=".65", label="Rejected fit (diagnostic only)")
    ax.set(xlabel="Physical window start (ms)", ylabel="Conditional spin (RPM)", title="Finite sphere rotation; physical accuracy unvalidated")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(rows[0]["group"] + " | event timestamps, not video playback FPS", fontsize=14)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_video(rows, markers, folder, destination, ball_diameter_m=.040, preview_index=0):
    """25 FPS slow review; every frame retains its actual source timestamp."""
    width, height = 1280, 640
    by_time = defaultdict(list)
    for item in markers:
        by_time[item["timestamp_us"]].append(item)
    centers = np.array([[r["center_x_px"], r["center_y_px"]] for r in rows])
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    span = np.maximum(hi - lo, 1)
    scale = min(650 / span[0], 285 / span[1])
    mapped = (centers - (lo + hi) / 2) * scale + [900, 260]
    background = Image.new("RGB", (width, height), "#141c28")
    draw = ImageDraw.Draw(background)
    draw.text((20, 8), rows[0]["group"] + " | marker tracking + ball translation + spin", font=font(24), fill="white")
    draw.text((20, 43), f"Conditional estimates | {ball_diameter_m*1000:g} mm sphere assumption | NOT validated 3D speed / physical RPM", font=font(17), fill="#ffd477")
    draw.rectangle((555, 80, 1255, 425), outline="#758297")
    draw.text((565, 86), "Ball-centre path in sensor coordinates (equal aspect)", font=font(17), fill="#d0d9e8")
    if len(mapped) > 1:
        draw.line([tuple(p) for p in mapped], fill="#4d596a", width=2)
    for i, row in enumerate(rows):
        if row["track_support"] != "ring_supported":
            p = mapped[i]; draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill="#c6a34c")
    draw.text((20, 613), "25 FPS review playback; time labels use source microseconds. Prediction-only marker points removed.", font=font(16), fill="#c4cfdf")
    log = destination.with_suffix(".encoding.log").open("wb")
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], stdin=subprocess.PIPE, stderr=log)
    trails = defaultdict(list)
    try:
        for i, row in enumerate(rows):
            frame = background.copy()
            with Image.open(folder / "binary" / row["image"]) as picture:
                frame.paste(picture.convert("RGB").resize((512, 512), Image.Resampling.NEAREST), (20, 80))
            d = ImageDraw.Draw(frame)
            for item in by_time[row["timestamp_us"]]:
                c = color(item["track_id"])
                xy = (20 + item["center_x_px"] * 2, 80 + item["center_y_px"] * 2)
                trail = trails[item["segment_id"]]
                trail.append(xy)
                if len(trail) > 1:
                    d.line(trail[-30:], fill=c, width=2)
                d.rectangle((20 + item["x1"]*2, 80 + item["y1"]*2, 20 + item["x2"]*2, 80 + item["y2"]*2), outline=c, width=2)
                d.text((xy[0]+3, xy[1]), f'T{item["track_id"]}', font=font(13), fill=c)
            p = mapped[i]
            d.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), fill="#ff8066")
            def val(value, unit, decimals=1):
                return f"{value:.{decimals}f} {unit}" if value is not None else "unavailable (insufficient evidence)"
            texts = [f'frame {row["frame_index"]}  |  t={row["timestamp_us"]} us',
                     f'Image speed: {val(row["image_speed_px_s"], "px/s")}',
                     f'Transverse estimate: {val(row["candidate_transverse_speed_mps"], "m/s", 2)}',
                     f'Spin window candidate: {val(row["window_candidate_rpm"], "RPM")}',
                     f'Ball support: {row["track_support"]}  |  eligible markers: {len(by_time[row["timestamp_us"]])}']
            for j, text in enumerate(texts):
                d.text((563, 440 + j*31), text, font=font(18), fill="#ecf1f8" if j != 3 else "#ffd477")
            if i == preview_index:
                frame.save(destination.with_name("poster.jpg"), quality=90)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
        log.close()
    if code:
        raise RuntimeError(f"video encoding failed: {destination}")


def build_report(dataset, tracking, spin_report, output, ball_diameter_m=.040,
                 calibration=None, master_camera=None, slave_offset_us=0, videos=True, runtime_report=None):
    dataset, tracking, spin_report, output = map(Path, (dataset, tracking, spin_report, output))
    # Validate the scientific-export boundary before any coordinate conversion.
    _, _, count = read_tracks(tracking / "rotation_tracks_all.csv", dataset)
    spin = json.loads(spin_report.read_text(encoding="utf-8"))
    if spin["input_tracks_sha256"] != sha256(tracking / "rotation_tracks_all.csv"):
        raise ValueError("spin report and tracking export are from different runs")
    rows, groups = ball_motion(dataset, ball_diameter_m)
    processed = {(p.parent.parent.name, p.parent.name) for p in tracking.glob("*/*/config.json")}
    if not processed:
        with (tracking / "rotation_tracks_all.csv").open(encoding="utf-8-sig", newline="") as stream:
            processed = {(r["camera"], r["group"]) for r in csv.DictReader(stream)}
    rows = [r for r in rows if (r["camera"], r["group"]) in processed]
    groups = [r for r in groups if (r["camera"], r["group"]) in processed]
    output.mkdir(parents=True, exist_ok=False)
    with (tracking / "rotation_tracks_all.csv").open(encoding="utf-8-sig", newline="") as stream:
        markers = list(csv.DictReader(stream))
    ball_index = {(r["camera"], r["group"], r["timestamp_us"]): r for r in rows}
    marker_groups = defaultdict(list)
    projected_markers = []
    for m in markers:
        for key in ["timestamp_us", "frame_index", "track_id"]:
            m[key] = int(m[key])
        for key in ["center_x_px", "center_y_px", "x1", "y1", "x2", "y2"]:
            m[key] = float(m[key])
        ball = ball_index[(m["camera"], m["group"], m["timestamp_us"])]
        meta = json.loads((dataset / "unlabeled_inference" / m["camera"] / m["group"] / "metadata.json").read_text())
        scale = ball["radius_px"] * float(meta["crop_radius_scale"]) / 128
        projected_markers.append({"camera": m["camera"], "group": m["group"], "segment_id": m["segment_id"],
                                  "timestamp_us": m["timestamp_us"], "frame_index": m["frame_index"],
                                  "marker_sensor_x_px": ball["center_x_px"] + (m["center_x_px"] - 128) * scale,
                                  "marker_sensor_y_px": ball["center_y_px"] + (m["center_y_px"] - 128) * scale,
                                  "crop_to_sensor_scale": scale, "source": m["source"],
                                  "geometry_verified": False})
        marker_groups[m["group"]].append(m)
    plot_groups = []
    for summary in groups:
        name, camera = summary["group"], summary["camera"]
        current = [r for r in rows if r["group"] == name]
        windows = [w for w in spin["windows"] if w["group"] == name]
        attach_spin(current, windows)
        folder = output / name
        folder.mkdir()
        meta = json.loads((dataset / "unlabeled_inference" / camera / name / "metadata.json").read_text())
        group_plot(current, marker_groups[name], windows, folder / "summary.png", 128 / float(meta["crop_radius_scale"]))
        counts = defaultdict(int)
        for marker in marker_groups[name]:
            counts[marker["timestamp_us"]] += 1
        preview_index = max(range(len(current)), key=lambda i: (counts[current[i]["timestamp_us"]], -abs(i - len(current) / 2)))
        if videos:
            render_video(current, marker_groups[name], dataset / "unlabeled_inference" / camera / name, folder / "motion.mp4", ball_diameter_m, preview_index)
        values = [w["candidate_spin_rpm"] for w in windows if w["numerically_usable"]]
        summary.update(candidate_spin_median_rpm=float(np.median(values)) if values else None,
                       usable_spin_windows=len(values), total_spin_windows=len(windows),
                       marker_observations=len(marker_groups[name]), spin_rpm=None)
        plot_groups.append({**summary, "video": f"{name}/motion.mp4" if videos else None,
                            "poster": f"{name}/poster.jpg" if videos else None, "preview_frame": preview_index,
                            "plot": f"{name}/summary.png", "rows": current})
        print("MOTION", name, len(current), "frames", flush=True)
    stereo = stereo_motion(rows, calibration, master_camera, slave_offset_us) if calibration else []
    write_csv(output / "ball_motion.csv", rows)
    write_csv(output / "group_summary.csv", groups)
    write_csv(output / "marker_tracks_sensor.csv", projected_markers)
    write_csv(output / "stereo_motion.csv", stereo, fields=["serve", "timestamp_us", "frame_index", "x_m", "y_m", "z_m", "reprojection_error_px", "master_camera", "candidate_3d_speed_mps", "speed_mps", "physical_measurement_validated"])
    runtime = json.loads(Path(runtime_report).read_text()) if runtime_report and Path(runtime_report).is_file() else None
    report = {"status": "motion_and_conditional_spin_visualized", "ball_samples": len(rows),
              "ball_velocity_samples": sum(r["velocity_usable"] for r in rows),
              "marker_observations": count, "groups": groups, "ball_diameter_assumption_m": ball_diameter_m,
              "spin_windows": spin["window_count"], "usable_spin_windows": spin["usable_window_count"],
              "calibration": str(calibration) if calibration else None, "stereo_samples": len(stereo),
              "stereo_master_camera": master_camera, "stereo_translation_unit": "metres" if calibration else None,
              "slave_timestamp_offset_added_us": slave_offset_us, "runtime": runtime,
              "physical_measurement_validated": False, "speed_mps": None, "spin_rpm": None,
              "tracking_sha256": sha256(tracking / "rotation_tracks_all.csv"), "spin_report_sha256": sha256(spin_report),
              "limitations": ["Pixel velocity uses source candidate ball centres; no independent ball-centre truth.",
                              "Diameter-scaled speed is an orthographic transverse estimate, not full 3D speed.",
                              "Ball diameter is a configurable assumption, not a measured value.",
                              "Historical synchronization evidence does not validate this training recording.",
                              "No real calibration means no stereo speed; no synthetic calibration is substituted.",
                              "Spin is a window candidate, not an independent per-frame measurement."],
              "review": "review.html"}
    (output / "motion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if runtime:
        (output / "sdk_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    _write_page(plot_groups, report, output)
    if stereo:
        _stereo_plot(stereo, output / "stereo_trajectory.png")
    return report


def _stereo_plot(rows, output):
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(projection="3d")
    for serve in sorted({r["serve"] for r in rows}):
        points = np.array([[r["x_m"], r["y_m"], r["z_m"]] for r in rows if r["serve"] == serve])
        ax.scatter(*points.T, s=8, label=serve)
    ax.set(xlabel="Master x (m)", ylabel="Master y (m)", zlabel="Master z (m)", title="Calibrated 3D candidates; accuracy unvalidated")
    ax.legend(); fig.savefig(output, dpi=140); plt.close(fig)


def _write_page(groups, report, output):
    # Embedded data permits file:// review without a web server or external CDN.
    data = json.dumps(groups, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = "PingPong 轨迹、球速与转速审核"
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>''' + title + '''</title>
<style>body{font:16px system-ui;background:#0e1724;color:#dfe8f6;max-width:1380px;margin:26px auto;padding:0 20px}a{color:#8ec6ff}.notice{background:#34301c;padding:16px;border-left:4px solid #edbf5a;line-height:1.6}select,button{background:#24364e;color:white;border:1px solid #637b9b;padding:9px;font:inherit;border-radius:5px}video,img{width:100%;border-radius:8px;background:#141c28}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#182539;padding:16px;border-radius:10px}.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}.kpi span{padding:12px;background:#20334b;border-radius:8px}canvas{width:100%;background:#111e30;border-radius:6px}input{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #34465d;text-align:left}@media(max-width:900px){.grid{grid-template-columns:1fr}}small{color:#adc0d8}</style>
<h1>''' + title + '''</h1>
<p class="notice">本页使用原始微秒时间戳。像素球速可直接由球心轨迹计算；m/s 是假设球径 40 mm、局部弱透视下的横向速度估计，不能当作完整三维球速。转速是通过数值检查的球面旋转候选，尚无独立真值验证。无真实标定时不生成三维球速。</p>
<p><a href="motion_report.json">完整报告</a> · <a href="ball_motion.csv">球心/球速 CSV</a> · <a href="marker_tracks_sensor.csv">标记轨迹原图坐标 CSV</a> · <a href="group_summary.csv">分组汇总</a> · <a href="stereo_motion.csv">三维候选 CSV（须标定）</a> · <a href="../spin/review.html">转速拟合细节</a></p>
<label>选择相机与轨迹 <select id="group"></select></label><div class="kpi" id="kpi"></div>
<div class="grid"><div class="card"><video id="video" controls preload="metadata"></video><p><small>默认展示有标记观测的示例帧。视频为 25 FPS 慢放；视频秒数与采集秒数不同。滑块显示真实采集时间。只显示连续三帧以上、有观测支持的标记段。</small></p><input id="slider" type="range" min="0" value="0"><p id="current"></p></div>
<div class="card"><canvas id="trajectory" width="760" height="330"></canvas><canvas id="timeline" width="760" height="250"></canvas><p><small>上图为原传感器球心位置，红点表示当前帧；下图蓝色为像素球速，橙色为短窗口转速候选，分别归一化显示，数值与单位见当前帧信息。空值不补成零。</small></p></div></div>
<h2>完整轨迹与曲线</h2><img id="plot"><p><a id="download">打开原始 PNG</a></p>
<h2>各组描述统计</h2><p>中位数仅汇总有效候选；不同相机可见时段不同，不直接合并。不报告实物准确率。</p><table><thead><tr><th>组</th><th>像素速度中位数 px/s</th><th>横向估计 m/s</th><th>转速候选中位 RPM</th><th>可用/全部窗口</th></tr></thead><tbody id="summary"></tbody></table>
<script>const groups=''' + data + ''';
const el=id=>document.getElementById(id);let active=0;
function num(x,n=1){return x===null||x===undefined?'—':Number(x).toFixed(n)}
function trajectory(g,i){const c=el('trajectory'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const rs=g.rows;const xs=rs.map(r=>r.center_x_px),ys=rs.map(r=>r.center_y_px);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);const scale=Math.min(650/Math.max(x1-x0,1),230/Math.max(y1-y0,1));const map=r=>[380+(r.center_x_px-(x0+x1)/2)*scale,180+(r.center_y_px-(y0+y1)/2)*scale];x.fillStyle='#cad9ee';x.font='16px system-ui';x.fillText('球心轨迹：原传感器坐标（等比例）',20,25);x.strokeStyle='#55748e';x.beginPath();rs.forEach((r,k)=>{const p=map(r);k?x.lineTo(...p):x.moveTo(...p)});x.stroke();const p=map(rs[i]);x.fillStyle='#ff826c';x.beginPath();x.arc(...p,6,0,Math.PI*2);x.fill();x.fillStyle='#abc1d9';x.fillText(`x=${num(rs[i].center_x_px)} px  y=${num(rs[i].center_y_px)} px`,20,310)}
function timeline(g,i){const c=el('timeline'),x=c.getContext('2d'),rs=g.rows;x.clearRect(0,0,c.width,c.height);const end=Math.max(rs[rs.length-1].seconds_since_group_start,1e-9);for(const [key,col] of [['image_speed_px_s','#69b9ff'],['window_candidate_rpm','#ffc464']]){const max=Math.max(1,...rs.filter(r=>r[key]!==null).map(r=>r[key]));x.strokeStyle=col;x.beginPath();let fresh=true;rs.forEach((r,k)=>{if(k&&(r.timestamp_us-rs[k-1].timestamp_us>750||r.frame_index!==rs[k-1].frame_index+1))fresh=true;if(r[key]===null){fresh=true;return}const p=[30+700*r.seconds_since_group_start/end,215-170*r[key]/max];fresh?x.moveTo(...p):x.lineTo(...p);fresh=false});x.stroke()}x.strokeStyle='#fff';const cursor=30+700*rs[i].seconds_since_group_start/end;x.beginPath();x.moveTo(cursor,30);x.lineTo(cursor,220);x.stroke();x.fillStyle='#69b9ff';x.font='15px system-ui';x.fillText('像素球速 px/s',30,20);x.fillStyle='#ffc464';x.fillText('窗口转速候选 RPM',230,20);x.fillStyle='#abc1d9';x.fillText('采集时间 '+num(end*1000)+' ms（曲线各自归一化）',30,245)}
function update(i,seek){const g=groups[active];i=Math.max(0,Math.min(g.rows.length-1,i));const r=g.rows[i];el('slider').value=i;el('current').textContent=`帧 ${r.frame_index} · t=${r.timestamp_us} μs · 球速 ${num(r.image_speed_px_s)} px/s · 横向估计 ${num(r.candidate_transverse_speed_mps,2)} m/s · 窗口转速 ${num(r.window_candidate_rpm)} RPM · ${r.track_support}`;trajectory(g,i);timeline(g,i);if(seek&&g.video)el('video').currentTime=i/25}
function select(){active=Number(el('group').value);const g=groups[active];el('video').style.display=g.video?'block':'none';if(g.video){el('video').poster=g.poster;el('video').onloadedmetadata=()=>update(g.preview_frame,true);el('video').src=g.video}el('plot').src=g.plot;el('download').href=g.plot;el('slider').max=g.rows.length-1;el('kpi').innerHTML=`<span>${g.frames} 帧 / ${g.marker_observations} 个标记观测</span><span>像素球速中位数 ${num(g.image_speed_median_px_s)} px/s</span><span>横向估计 ${num(g.candidate_transverse_speed_median_mps,2)} m/s</span><span>转速候选 ${num(g.candidate_spin_median_rpm)} RPM</span>`;update(g.preview_frame||0,false)}
groups.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=g.group;el('group').append(o);const tr=document.createElement('tr');for(const value of [g.group,num(g.image_speed_median_px_s),num(g.candidate_transverse_speed_median_mps,2),num(g.candidate_spin_median_rpm),g.usable_spin_windows+'/'+g.total_spin_windows]){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('summary').append(tr)});
el('group').onchange=select;el('slider').oninput=()=>update(Number(el('slider').value),true);el('video').ontimeupdate=()=>update(Math.round(el('video').currentTime*25),false);if(groups.length)select();</script></html>'''
    page = page.replace("假设球径 40 mm", f'假设球径 {report["ball_diameter_assumption_m"]*1000:g} mm')
    runtime = report.get("runtime") or {}
    status = f'<p>SDK 导入：{"通过" if runtime.get("sdk_import_ok") else "未核验"}；RAW 小窗口重放：{"通过" if runtime.get("raw_replay_ok") else "未核验"}；已发现相机：{len(runtime.get("devices", []))}；真实标定：{"已提供" if report["calibration"] else "未提供"}。实机采集未验证。</p>'
    page = page.replace("<h2>完整轨迹与曲线</h2>", status + "<h2>完整轨迹与曲线</h2>")
    (output / "review.html").write_text(page, encoding="utf-8")
