"""Local, portable HTML and scientific plots for reviewing conditional spin."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .marker_spin import read_tracks, make_pairs
from .spin import _rotate_vectors


def create_review(report, tracks, dataset, output):
    output, dataset = Path(output), Path(dataset)
    segments, _, _ = read_tracks(tracks, dataset)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax, camera in zip(axes, ("F3E", "F59")):
        groups = sorted({w["group"] for w in report["windows"] if w["camera"] == camera})
        for group in groups:
            rows = [w for w in report["windows"] if w["group"] == group and w["numerically_usable"]]
            if rows:
                ax.scatter([w["seconds_since_first"] * 1000 for w in rows],
                           [w["candidate_spin_rpm"] for w in rows], label=group, s=24)
        ax.set(title=f"{camera}: conditional estimates (not validated physical RPM)",
               xlabel="Window start since first track observation (ms)", ylabel="Candidate RPM")
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, ncol=4)
    fig.savefig(output / "spin_candidates.png", dpi=150)
    plt.close(fig)
    galleries = []
    residuals = []
    for camera, group in sorted({(w["camera"], w["group"]) for w in report["windows"]}):
        usable = [w for w in report["windows"] if w["camera"] == camera and w["group"] == group and w["numerically_usable"]]
        if not usable:
            continue
        # Display the window with the most observed points; this is a review
        # choice, not checkpoint selection or a claim of validation.
        selected = max(usable, key=lambda w: (w["unique_observations"], w["inlier_fraction"]))
        runs = [(segment, run) for (cam, name, segment), run in segments.items() if (cam, name) == (camera, group)]
        radius = runs[0][1][0]["radius"]
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        visible = [(segment, [r for r in run if selected["start_us"] <= r["timestamp"] < selected["end_us"]]) for segment, run in runs]
        middle = min((r for _, run in visible for r in run), key=lambda r: abs(r["timestamp"] - (selected["start_us"] + selected["end_us"]) / 2))
        image = dataset / "unlabeled_inference" / camera / group / "binary" / Path(middle["image"]).name
        with Image.open(image) as img:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=.65)
        ax.add_patch(plt.Circle((128, 128), radius, fill=False, color="dodgerblue", linewidth=1))
        ax.add_patch(plt.Circle((128, 128), radius - 12, fill=False, color="dodgerblue", linestyle="--", linewidth=.7))
        for segment, run in visible:
            run = [r for r in run if r["track_support"] == "ring_supported"]
            if len(run) < 2:
                continue
            measured = np.array([r["point"] for r in run])
            times = np.array([(r["timestamp"] - run[0]["timestamp"]) * 1e-6 for r in run])
            predicted = _rotate_vectors(np.tile(run[0]["sphere"], (len(run), 1)),
                                        np.array(selected["omega_camera_rad_s"]), times)[:, :2] * radius + 128
            ax.plot(measured[:, 0], measured[:, 1], "o-", color="lime", markersize=2, linewidth=1)
            ax.plot(predicted[:, 0], predicted[:, 1], ".--", color="tomato", markersize=2, linewidth=1)
            ax.text(*measured[0], segment.split("/")[-2], color="cyan", fontsize=7)
        ax.set(xlim=(0, 255), ylim=(255, 0), xlabel="Crop x (px)", ylabel="Crop y (px)",
               title=f"{group}: candidate {selected['candidate_spin_rpm']:.0f} RPM\nGreen: observed; red: model from segment anchor\nNot validated physical spin; window drift is visible")
        filename = f"{group}_reprojection.png"
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        galleries.append(f'<figure><img src="{filename}" alt="{group} reprojection"><figcaption>{group}</figcaption></figure>')
    for w in report["windows"]:
        if "omega_camera_rad_s" not in w:
            continue
        runs = [(segment, run) for (cam, group, segment), run in segments.items() if (cam, group) == (w["camera"], w["group"])]
        pairs = make_pairs(runs, w["start_us"], w["start_us"] + report["config"]["window_us"], report["config"]["maximum_pair_us"])
        if not pairs:
            continue
        pred = _rotate_vectors(np.array([p[1]["sphere"] for p in pairs]), np.array(w["omega_camera_rad_s"]), np.array([p[3] for p in pairs]))
        for p, predicted in zip(pairs, pred):
            segment, a, b, dt = p
            xy = predicted[:2] * a["radius"] + 128
            residuals.append(dict(window_id=w["window_id"], segment_id=segment, start_us=a["timestamp"],
                                  end_us=b["timestamp"], dt_us=round(dt * 1e6),
                                  observed_x_px=b["point"][0], observed_y_px=b["point"][1],
                                  predicted_x_px=xy[0], predicted_y_px=xy[1],
                                  error_px=float(np.linalg.norm(xy - b["point"])),
                                  numerically_usable_window=w["numerically_usable"]))
    fields = ["window_id", "segment_id", "start_us", "end_us", "dt_us", "observed_x_px", "observed_y_px", "predicted_x_px", "predicted_y_px", "error_px", "numerically_usable_window"]
    with (output / "spin_reprojection.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(residuals)
    table = []
    for w in report["windows"]:
        candidate = f'{w["candidate_spin_rpm"]:.1f}' if w["candidate_spin_rpm"] is not None else "—"
        reason = ", ".join(w["rejection_reasons"]) or "数值检查通过；物理测量未验证"
        error = f'{w["inlier_reprojection_rms_px"]:.2f}' if "inlier_reprojection_rms_px" in w else "—"
        table.append(f'<tr><td>{html.escape(w["group"])}</td><td>{w["seconds_since_first"] * 1000:.1f}</td><td>{candidate}</td><td>{w["pairs"]}</td><td>{error}</td><td>{html.escape(reason)}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PingPong 转速审核</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f4f5f7;color:#18212c}img{max-width:100%}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}table{width:100%;border-collapse:collapse;background:white;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}.note{background:#fff0cb;padding:16px}a{color:#1258a4}</style>
<h1>PingPong 离线转速审核</h1><p class="note">此页输出的是在球面、时间戳和标记点身份假设下的转速候选值。SDK 时间、球心/半径与真实转速尚未独立验证，spin_rpm 保持空值。数值检查通过不等于实际测量准确。两个相机分别计算，未作双目融合。</p>
'''
    page += f'<p>输入：{report["input_observations"]} 个观测，{report["input_segments"]} 段连续轨迹；数值可用窗口 {report["usable_window_count"]}/{report["window_count"]}。每个输入段至少连续三帧，预测补点不参与求解；额外排除球体回退帧。</p>'
    page += '<p><a href="spin_report.json">完整报告</a> · <a href="spin_by_group.csv">分组汇总 CSV</a> · <a href="spin_windows.csv">逐窗口 CSV</a> · <a href="spin_reprojection.csv">逐对应点残差 CSV</a></p>'
    page += '<h2>分相机、分轨迹的描述统计</h2><p>中位数和范围仅描述数值可用的重叠窗口；不是测量置信区间。两相机覆盖时段可能不同，不直接平均。</p><table><tr><th>组</th><th>可用/全部窗口</th><th>候选中位 RPM</th><th>候选范围 RPM</th></tr>'
    for row in report['group_summaries']:
        median = f'{row["candidate_median_rpm"]:.1f}' if row['candidate_median_rpm'] is not None else '—'
        span = f'{row["candidate_min_rpm"]:.1f}–{row["candidate_max_rpm"]:.1f}' if row['candidate_min_rpm'] is not None else '—'
        page += f'<tr><td>{html.escape(row["group"])}</td><td>{row["usable_windows"]}/{row["windows"]}</td><td>{median}</td><td>{span}</td></tr>'
    page += '</table><h2>旋转方向与时间稳定性</h2><p>有向转轴使用各自相机坐标：x 右、y 下、z 远离相机；正负号保留。前表面箭头是可见球心处标记的运动方向。上旋/下旋/侧旋的体育分类需要飞行方向和世界竖直方向，当前不作猜测。工程稳定表示至少三个窗口符合 15° 轴向和 20% 转速一致性检查，不是精度验证。</p><table><tr><th>组</th><th>状态</th><th>稳定窗口</th><th>工程候选 RPS</th><th>有向单位轴 (x,y,z)</th><th>前表面运动</th></tr>'
    labels = {'up': '↑ 上', 'down': '↓ 下', 'left': '← 左', 'right': '→ 右', 'up_left': '↖ 左上', 'up_right': '↗ 右上', 'down_left': '↙ 左下', 'down_right': '↘ 右下', 'unresolved': '未确定'}
    for row in report['group_summaries']:
        axis = ', '.join(f'{row[k]:+.3f}' for k in ['axis_x','axis_y','axis_z']) if row.get('axis_x') is not None else '—'
        rate = f'{row["engineering_candidate_rps"]:.1f}' if row.get('engineering_candidate_rps') is not None else '—'
        page += f'<tr><td>{html.escape(row["group"])}</td><td>{row.get("engineering_direction_status", "not_checked")}</td><td>{row.get("stable_windows", 0)}</td><td>{rate}</td><td>{axis}</td><td>{labels.get(row.get("front_motion_label"), "未确定")}</td></tr>'
    page += '</table><img src="spin_candidates.png" alt="spin candidates"><h2>观测与旋转模型重投影</h2><p>每组展示一个观测点最多的可用窗口。绿色为跟踪框中心，红色为从每段起点按拟合角速度预测的位置；观察长时间漂移与错配。背景仅为窗口中一张裁剪图。</p><div class="gallery">'
    page += "".join(galleries) + '</div><h2>全部窗口</h2><table><thead><tr><th>组</th><th>相对起点 ms</th><th>候选 RPM</th><th>对应点对</th><th>内点 RMS px</th><th>检查结果</th></tr></thead><tbody>'
    page += "".join(table) + "</tbody></table></html>"
    (output / "review.html").write_text(page, encoding="utf-8")
