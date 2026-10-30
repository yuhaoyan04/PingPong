"""Spin-first engineering review: magnitude context, signed axes, event timing."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .motion import write_csv
from .spin_direction import describe_omega, summarize_directions, classify_in_reference


SOURCES = [
    {"id": "spinsight_guidance", "url": "https://spinsight.com/app/experimental/help/",
     "evidence_type": "measurement_vendor_usage_guidance_not_population_study",
     "amateur_topspin_serve_rps": [20, 60], "amateur_backspin_serve_rps": [20, 40],
     "slow_serve_kmh": [5, 15], "fast_serve_kmh": [20, 30]},
    {"id": "yoshida_2014", "url": "https://www.jstage.jst.go.jp/article/jjpehss/59/1/59_13068/_article/-char/en",
     "evidence_type": "primary_peer_reviewed_world_championship_service_study", "services": 329,
     "observed_rps": [13.7, 62.5], "male_mean_sd_rps": [46.0, 9.0], "female_mean_sd_rps": [39.2, 9.3]},
    {"id": "tamaki_yoshida_2025", "url": "https://journals.sagepub.com/doi/pdf/10.1177/22150218251409592?download=true",
     "evidence_type": "primary_peer_reviewed_professional_T_League_services", "services": 1773,
     "male_short_median_iqr_rps": [46.4, 15.8], "female_short_median_iqr_rps": [38.9, 13.6],
     "male_long_median_iqr_rps": [50.9, 13.8], "female_long_median_iqr_rps": [47.6, 14.7]},
    {"id": "dvsense_timestamp", "url": "https://sdk.dvsense.com/en/html/intro_to_dvs_en.html",
     "timestamp_storage": "uint64", "timestamp_unit": "microseconds"},
    {"id": "evt3_spec", "url": "https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html",
     "timestamp_quantum_us": 1, "time_high_low_bits": [12, 12], "wrap_period_us": 16777216},
]


LABELS = {"up": "↑ 上", "down": "↓ 下", "left": "← 左", "right": "→ 右",
          "up_left": "↖ 左上", "up_right": "↗ 右上", "down_left": "↙ 左下", "down_right": "↘ 右下", "unresolved": "未确定"}
STATUSES = {"stable_candidate": "方向与转速自洽", "insufficient_windows": "窗口不足",
            "inconsistent_candidate": "方向/转速变化需复核", "no_candidate": "无可用候选"}


def build_acceptance(spin_path, motion_csv, timing_paths, output, orientation_profile=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spin = json.loads(Path(spin_path).read_text(encoding="utf-8"))
    timing = [json.loads(Path(p).read_text(encoding="utf-8")) for p in timing_paths]
    orientation = json.loads(Path(orientation_profile).read_text()) if orientation_profile else None
    summaries, detailed = [], []
    for original in spin["group_summaries"]:
        windows = [w for w in spin["windows"] if w["group"] == original["group"]]
        summary = {**original, **summarize_directions(windows)}
        rates = [w["candidate_spin_rpm"] / 60 for w in windows if w["numerically_usable"]]
        summary["all_usable_median_rps"] = float(np.median(rates)) if rates else None
        summary["all_usable_min_rps"] = min(rates) if rates else None
        summary["all_usable_max_rps"] = max(rates) if rates else None
        summary["within_2009_observed_serve_examples"] = all(13.7 <= r <= 62.5 for r in rates) if rates else None
        summary["very_high_serve_warning"] = any(r > 120 for r in rates)
        summary["range_check_is_accuracy_validation"] = False
        excluded = {r["window_id"] for r in summary["excluded_windows"]}
        for w in windows:
            row = dict(w)
            if w["numerically_usable"]:
                row.update(describe_omega(w["omega_camera_rad_s"]))
            row["temporal_outlier_for_group_consensus"] = w["window_id"] in excluded
            detailed.append(row)
        if orientation and summary["engineering_stable"]:
            rotation = np.asarray(orientation["camera_to_reference"][original["camera"]], float)
            if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
                raise ValueError("camera_to_reference must be a measured proper rotation")
            axis = np.array([summary[k] for k in ["axis_x", "axis_y", "axis_z"]])
            omega = rotation @ axis * summary["engineering_candidate_rps"] * 2 * np.pi
            summary.update(classify_in_reference(omega, orientation["flight_reference_by_group"][original["group"]], orientation["up_reference"]))
        summaries.append(summary)
    with Path(motion_csv).open(encoding="utf-8-sig", newline="") as stream:
        speeds = [float(r["candidate_transverse_speed_mps"]) for r in csv.DictReader(stream) if r["candidate_transverse_speed_mps"]]
    good_rates = [w["candidate_spin_rpm"] / 60 for w in detailed if w["numerically_usable"]]
    states = Counter(r["engineering_direction_status"] for r in summaries)
    report = {"priority": "spin_rate_and_signed_direction", "magnitude_units": ["rps", "rpm"],
              "input_spin_report": str(Path(spin_path).resolve()), "sources": SOURCES,
              "groups": summaries, "group_status_counts": dict(states), "windows": detailed,
              "usable_window_rps_range": [min(good_rates), max(good_rates)] if good_rates else None,
              "engineering_thresholds": {"minimum_windows": 3, "axis_medoid_limit_deg": 15,
                                          "relative_rate_limit": .20, "required_consensus_fraction": .60,
                                          "serve_warning_above_rps": 120},
              "event_timing_reports": timing,
              "all_current_raw_decoded_intervals_verified": bool(timing and all(t["decoded_intervals_verified"] for t in timing)),
              "historical_raw_mismatch_count": sum(t.get("current_file_matches_historical_record") is False for t in timing),
              "ball_speed_secondary": {"samples": len(speeds), "candidate_transverse_mps_min_median_max":
                                       [float(np.min(speeds)), float(np.median(speeds)), float(np.max(speeds))] if speeds else None,
                                       "gross_outlier_warning_count_above_20mps": sum(s > 20 for s in speeds),
                                       "values_clipped": False, "is_full_3d_speed": False},
              "orientation_profile": str(orientation_profile) if orientation_profile else None,
              "camera_coordinate_frame": "x right, y down, z away; right-hand signed angular velocity",
              "front_surface_point": [0, 0, -1], "front_surface_velocity_formula": "(-omega_y, omega_x)",
              "training_raw_event_timing_verified": False, "physical_accuracy_validated": False,
              "reference_ranges_used_to_modify_estimates": False,
              "limitations": ["Literature overlap is a magnitude check, not evidence of measurement accuracy.",
                              "Camera axes cannot be averaged across cameras without extrinsics.",
                              "Top/back/side sport labels require flight/up and camera-to-reference mapping.",
                              "Temporal medoid screening preserves every raw window and may flag real post-impact changes.",
                              "1 us encoded quantization is not a calibrated bound on physical timing accuracy.",
                              "The audited clock-test files differ from historical reports and are not the ball-training RAW."]}
    (output / "acceptance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    fields = ["camera", "group", "all_usable_median_rps", "all_usable_min_rps", "all_usable_max_rps", "engineering_direction_status", "engineering_candidate_rps", "engineering_candidate_rpm", "stable_windows", "direction_windows", "axis_x", "axis_y", "axis_z", "front_motion_label", "image_z_spin_sense", "all_windows_max_axis_deviation_deg", "stable_windows_max_axis_deviation_deg", "within_2009_observed_serve_examples", "world_spin_type"]
    with (output / "spin_direction_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(summaries)
    window_fields = ["camera", "group", "window_id", "start_us", "end_us", "numerically_usable", "candidate_rps", "candidate_spin_rpm", "axis_x", "axis_y", "axis_z", "front_direction_dx", "front_direction_dy", "front_motion_label", "image_z_spin_sense", "temporal_outlier_for_group_consensus"]
    with (output / "spin_direction_windows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=window_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(detailed)
    _plots(report, output)
    _html(report, output, spin_path, timing_paths)
    return report


def _plots(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, camera in zip(axes, ["F3E", "F59"]):
        rows = [g for g in report["groups"] if g["camera"] == camera]
        ax.axvspan(20, 60, color="tab:blue", alpha=.09, label="Vendor amateur serve guidance (20-60 rps)")
        ax.axvline(46.4, color="tab:red", linestyle=":", label="Professional male short-serve median (46.4)")
        ax.axvline(38.9, color="tab:purple", linestyle=":", label="Professional female short-serve median (38.9)")
        for i, r in enumerate(rows):
            if r["all_usable_median_rps"] is None:
                continue
            centre = r["all_usable_median_rps"]
            ax.errorbar(centre, i, xerr=np.array([[centre-r["all_usable_min_rps"]], [r["all_usable_max_rps"]-centre]]),
                        fmt="o", color="tab:green" if r["engineering_stable"] else "tab:orange", capsize=4)
        ax.set(yticks=range(len(rows)), yticklabels=[r["group"] for r in rows], xlabel="Original window range / median spin (rps)", title=f"{camera} | no clipping to reference ranges")
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc="lower right")
    fig.savefig(output / "spin_reference_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 7, figsize=(15, 5.3), constrained_layout=True)
    for row, camera in enumerate(["F3E", "F59"]):
        groups = [g for g in report["groups"] if g["camera"] == camera]
        for ax, group in zip(axes[row], groups):
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=".7"))
            ax.axhline(0, color=".9", linewidth=.5); ax.axvline(0, color=".9", linewidth=.5)
            if group["engineering_stable"] and group["front_direction_dx"] is not None:
                x, y = group["front_direction_dx"], group["front_direction_dy"]
                ax.annotate("", xy=(.8*x, -.8*y), xytext=(-.3*x, .3*y), arrowprops={"arrowstyle": "->", "lw": 3, "color": "tab:green"})
                axis = f'axis ({group["axis_x"]:+.2f}, {group["axis_y"]:+.2f}, {group["axis_z"]:+.2f})'
                ax.set_xlabel(axis, fontsize=7)
            else:
                ax.text(0, 0, "not stable /\ninsufficient", ha="center", fontsize=8)
            ax.set(title=group["group"], xlim=(-1.2,1.2), ylim=(-1.2,1.2), xticks=[], yticks=[])
            ax.set_aspect("equal")
    fig.suptitle("Visible front-centre motion arrows in each camera view (NOT world top/back/side labels)")
    fig.savefig(output / "spin_direction_arrows.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, timing in zip(axes, report["event_timing_reports"]):
        histogram = Path(timing["raw"])  # title is sourced from header, not inferred roles
        serial = next((h.split()[-1] for h in timing["header"] if h.startswith("% serial_number ")), "camera")
        stats = timing["sdk"]
        # Quantile indicators show integer decoded intervals, not frame times.
        labels = ["minimum positive", "median", "95th percentile", "99th percentile", "maximum"]
        values = [stats["minimum_positive_delta_us"], stats["delta_quantiles_us"]["0.5"], stats["delta_quantiles_us"]["0.95"], stats["delta_quantiles_us"]["0.99"], stats["maximum_delta_us"]]
        ax.bar(labels, values, color="tab:blue")
        ax.tick_params(axis="x", labelrotation=22)
        ax.set(title=f'{serial}: {stats["events"]:,} CD events', ylabel="Adjacent decoded interval (us)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "event_interval_summary.png", dpi=150); plt.close(fig)


def _html(report, output, spin_path, timing_paths):
    def num(x, digits=1):
        return "—" if x is None else f"{x:.{digits}f}"
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PingPong 转速、方向与事件时间审核</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 18px;background:#f3f5f8;color:#192332;line-height:1.6}img{max-width:100%}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.note{padding:15px;background:#fff0c9}a{color:#135aa2}code{background:#e6eaf0;padding:2px 5px}</style>
<h1>转速与方向优先审核</h1><p class="note">文献范围用于核对数量级，算法没有按范围截断或“拉回”结果。方向目前在各相机坐标内表达。工程自洽不等于实物精度已验证；体育意义的上旋/下旋/侧旋需要飞行方向与世界坐标映射。</p>'''
    page += '<p><a href="acceptance_report.json">完整审核 JSON</a> · <a href="spin_direction_summary.csv">分组转速与方向</a> · <a href="spin_direction_windows.csv">逐窗口原始方向</a> · <a href="'+html.escape(os.path.relpath(Path(spin_path).parent/'review.html', output))+'">拟合残差详情</a></p>'
    page += '<h2>发球转速参考</h2><table><tr><th>对象</th><th>RPS（转/秒）</th><th>RPM（转/分）</th><th>证据</th></tr>'
    benchmarks = [("业余上旋发球", "20–60", "1200–3600", SOURCES[0]["url"], "测量厂商经验范围"),
                  ("业余下旋发球", "20–40", "1200–2400", SOURCES[0]["url"], "测量厂商经验范围"),
                  ("职业短发球：男 / 女中位数", "46.4 / 38.9", "2784 / 2334", SOURCES[2]["url"], "1773 次 T 联赛发球研究"),
                  ("职业长发球：男 / 女中位数", "50.9 / 47.6", "3054 / 2856", SOURCES[2]["url"], "同一研究，长短发球分别统计"),
                  ("2009 世乒赛实测范围", "13.7–62.5", "822–3750", SOURCES[1]["url"], "329 次发球；样本范围不是物理上下限")]
    for title, rps, rpm, url, source in benchmarks:
        page += f'<tr><td>{title}</td><td>{rps}</td><td>{rpm}</td><td><a href="{html.escape(url)}">{source}</a></td></tr>'
    page += '</table><p>发球与拉冲分开比较；职业运动员也会选择弱旋转/不转发球。符合参考范围不能反推出测量准确或选手水平。</p><img src="spin_reference_comparison.png">'
    span = report["usable_window_rps_range"]
    range_label = f'{num(span[0])}–{num(span[1])} RPS' if span else '无可用候选'
    page += f'<p>当前可用窗口范围：{range_label}。方向状态统计：{html.escape(str(report["group_status_counts"]))}。</p>'
    page += '<h2>方向与稳定性</h2><p>有向单位轴 (x,y,z)：x 向画面右、y 向下、z 远离相机，按右手规则。前表面运动箭头由 <code>(−ωy, ωx)</code> 计算。至少三个窗口、轴向差 15°、转速相对差 20%、60% 共识为当前工程一致性阈值；保留全部原始窗口。撞台/触球也可能产生真实方向变化，需结合画面复核。</p><table><tr><th>组</th><th>原始中位 RPS</th><th>工程 RPS</th><th>有向轴</th><th>前表面运动</th><th>状态 / 一致窗口</th></tr>'
    for g in report["groups"]:
        axis = ", ".join(num(g[k], 3) for k in ["axis_x","axis_y","axis_z"]) if g["axis_x"] is not None else "—"
        page += f'<tr><td>{html.escape(g["group"])}</td><td>{num(g["all_usable_median_rps"])}</td><td>{num(g["engineering_candidate_rps"])}</td><td>{axis}</td><td>{LABELS[g["front_motion_label"]]}</td><td>{STATUSES[g["engineering_direction_status"]]} / {g["stable_windows"]}</td></tr>'
    page += '</table><img src="spin_direction_arrows.png"><h2>逐事件时间间隔</h2><p>直接计算 <code>int64(t[i]) − int64(t[i−1])</code>，跨 SDK 批次也连续相减。不排序、不删零、不把负数改成正数。时间戳单位为微秒，EVT3 编码量化为 1 μs；同一微秒内可有多个事件，因此 Δt=0 合法，不能解释为已证明物理上同时发生。</p><p><a href="'+SOURCES[3]["url"]+'">DVSense 时间戳定义</a> · <a href="'+SOURCES[4]["url"]+'">EVT3 时间编码</a></p><table><tr><th>文件 / 相机</th><th>事件数</th><th>最小正间隔 μs</th><th>中位 μs</th><th>零间隔比例</th><th>倒退数</th><th>独立解码逐事件一致</th></tr>'
    for timing, source_path in zip(report["event_timing_reports"], timing_paths):
        s = timing["sdk"]; url = html.escape(os.path.relpath(Path(source_path), output))
        page += f'<tr><td><a href="{url}">{html.escape(Path(timing["raw"]).name)}</a></td><td>{s["events"]:,}</td><td>{s["minimum_positive_delta_us"]}</td><td>{s["delta_quantiles_us"]["0.5"]}</td><td>{num(s["zero_fraction"]*100,2)}%</td><td>{s["negative_intervals"]}</td><td>{timing["sdk_reference_exact_agreement"]}</td></tr>'
        sample = html.escape(os.path.relpath(Path(source_path).parent/'adjacent_events_sample.csv',output))
        page += f'<tr><td colspan="7"><a href="{sample}">逐事件样例与批次边界 CSV</a>；同目录 events_*.npz 保存全部事件、Δt、有效掩码和全局事件索引。</td></tr>'
    page += '</table><img src="event_interval_summary.png"><p class="note">两份当前 RAW 副本与历史记录的事件总数和结束时刻不一致，不能直接沿用历史 PASS。独立 EVT3 解码与 SDK 完整事件序列一致，验证的是当前文件中的整数时间间隔。它们是同步测试数据，不是本批乒乓球发球 RAW；尚未证明原采集文件完整、硬件时钟准确或本批训练采集时间无异常。</p>'
    page += '<h2>时间层级与次要球速</h2><p>逐事件间隔随事件变化；标记裁剪序列通常每 500 μs 采样、每图积累 1000 μs 事件。转速的 Δt 使用同一标记两次观测的时间差，不使用全画面事件间隔的中位数。视频播放 FPS 仅影响慢放。</p>'
    speed = report["ball_speed_secondary"]["candidate_transverse_mps_min_median_max"]
    if speed:
        page += f'<p>球速保留为次要量：当前投影估计最小 / 中位 / 最大为 {num(speed[0],2)} / {num(speed[1],2)} / {num(speed[2],2)} m/s（球径 40 mm 假设），没有裁剪数值。当前数量级可用于工程参考，不当成精确三维球速。</p>'
    page += '</html>'
    (output / "review.html").write_text(page, encoding="utf-8")
