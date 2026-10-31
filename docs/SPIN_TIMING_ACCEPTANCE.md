# 转速、方向和事件时间优先验收

本轮将转速和有向旋转轴作为主要结果，球速只进行数量级核对。审核页：`output/latest_spin/acceptance/review.html`。研究范围用于校验，不参与修改拟合值；不将转速截断到“正常区间”。

后续逐次发球的双视角互补与零转速修复见 [逐次发球联合建模](SERVE_TWO_VIEW_SPIN.md)，新入口为 `output/latest_serve/review.html`。下文为保留的单视角与同步测试 RAW 审计；18.81 RPS 仅为该批可用结果的最小值，允许的转速下限为 0。

## 发球参考：RPS 与 RPM 分清

`RPM = RPS × 60`。发球与拉冲分开比较。

| 对象 | 转速参考（RPS，转/秒） | RPM，转/分 | 来源与适用范围 |
| --- | --- | --- | --- |
| 业余上旋发球 | 20–60 | 1200–3600 | [SPINSIGHT 官方使用指导](https://spinsight.com/app/experimental/help/)；厂商经验范围，并非人口抽样研究 |
| 业余下旋发球 | 20–40 | 1200–2400 | 同上；弱旋转和不转发球可以更低 |
| 世界级选手：男性均值±标准差 | 46.0±9.0 | 2760±540 | [Yoshida 等，2014](https://www.jstage.jst.go.jp/article/jjpehss/59/1/59_13068/_article/-char/en)，2009 世乒赛，329 次发球 |
| 同研究：女性均值±标准差 | 39.2±9.3 | 2352±558 | 样本实测范围为 13.7–62.5 RPS，不能当成所有发球的上下限 |
| 职业短发球：男/女中位数 | 46.4 / 38.9 | 2784 / 2334 | [Tamaki 与 Yoshida，2025](https://journals.sagepub.com/doi/pdf/10.1177/22150218251409592?download=true)，1773 次 T 联赛发球 |
| 同研究：职业长发球中位数 | 50.9 / 47.6 | 3054 / 2856 | 长短发球分别统计；不是每位职业选手都追求最大转速 |

职业发球与业余发球的范围重叠。比赛年代、球材质、测量时段和技术目的不同，不能用一个数字识别选手水平或证明算法精度。

当前 53 个数值可用窗口为 **18.81–49.96 RPS，即 1128–2998 RPM**，在常见发球数量级内。参考合理只是第一项检查，另需考察方向、时间连续性和拟合残差。

## 当前方向与工程一致性

输出保留有向单位转轴 `(axis_x, axis_y, axis_z)` 和角速度向量 `omega_camera_rad_s`。坐标为相机局部右手系：x 向右、y 向下、z 远离相机。反向旋转相差 180°，不能把 `ω` 和 `−ω` 当作同一方向。

可见前表面球心点是 `(0,0,−1)`，它的运动方向由 `ω × p = (−ωy, ωx, 0)` 得到，报告提供对应箭头。绕相机 z 轴的正分量在画面中为顺时针，但这只是绕视轴的旋转分量，不代表所有投影点都沿同一圆周运动。

当前工程一致性阈值：至少三个窗口；相对有向轴 medoid 的偏差不超过 15°；转速相对组中位数偏差不超过 20%；共识至少 60%。这些是筛选参数，不是实物测量误差界限。原始窗口及偏离记录全部保留，撞台或再次触球造成的真实变化同样可能被提示复核。

本批 14 组结果：**8 组方向与转速自洽，4 组窗口不足，1 组变化需复核，1 组没有数值可用候选**。`serve07_F59` 为变化需复核组；不能仅因它的 RPM 看起来合理就认定方向稳定。

体育意义的上旋/下旋/侧旋需要同一坐标系中的球飞行方向、竖直方向和相机姿态。当前不使用“画面向上”猜测世界竖直，也不直接平均两台相机的轴。代码提供 `classify_in_reference`；需要映射时可向 `spin-check` 指定实测 `--orientation-profile`：

```text
camera_to_reference: 每台相机的真实 3×3 旋转矩阵（正交且 det=+1）
up_reference: 共同参考系的向上三维向量
flight_reference_by_group: 每组在共同参考系的三维飞行方向
```

没有这些输入时，输出相机坐标中的方向和前表面运动箭头，`world_spin_type` 为空。它不依赖高精度球速大小，但需要正确的方向参考。

## 逐事件时间间隔

[DVSense 的数据格式文档](https://sdk.dvsense.com/en/html/intro_to_dvs_en.html) 定义事件时间戳为 uint64、单位微秒。[EVT3 规范](https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html) 给出低 12 位时间的 1 μs 分辨率及高低位组合规则。仓库中的 DVSLume 手册也说明微秒级时间分辨率，以及主从系统使用名义 1 MHz 时钟。

实际保存和计算：`delta_us[i] = int64(timestamp[i]) - int64(timestamp[i-1])`。先转有符号整数，避免 uint64 减法将时间倒退变成巨大的正间隔。跨 SDK 批次保留上一事件，零间隔合法，负间隔保留并报告，不排序、不删重、不用主机接收时刻替换。

当前两份同步测试 RAW 全量核验：

| 相机 | 当前解码事件数 | 最小正间隔 | 中位间隔 | 零间隔比例 | 时间倒退 |
| --- | --- | --- | --- | --- | --- |
| F59 | 3,790,694 | 1 μs | 8 μs | 4.02% | 0 |
| F3E | 3,525,708 | 1 μs | 10 μs | 3.28% | 0 |

独立按 EVT3 位布局解码原始字节，与 SDK 输出逐事件 `(x,y,p,t)` 的数量和完整序列 SHA-256 一致，包含向量事件展开和 24 位时间回绕。它验证当前文件的解码与整数间隔，不证明物理计时具有 1 μs 绝对准确度。相同时间戳只表示落在同一个微秒时间格，不能认定真实物理上同时发生。

**当前文件与历史记录不一致**：两份当前 RAW 都为 32,505,856 字节。F59 最后事件为 47,542,950 μs，而历史报告为 60,523,996 μs、5,070,797 个事件；F3E 当前为 54,445,538 μs，而历史为 60,523,991 μs、4,018,041 个事件。因此不能直接沿用历史完整性/同步 PASS，也不能用这些同步测试文件代替本批发球 RAW 的时间验证。

## 时间层级

| 时间量 | 本项目含义 |
| --- | --- |
| 1 μs | EVT3 时间戳量化单位，不是每两个事件都间隔 1 μs |
| 相邻事件 Δt | 逐事件实际相减，随场景变化，可能为 0、1、8、10 μs 等 |
| 500 μs | 当前标记裁剪序列的常见采样步长 |
| 1000 μs | 每张事件裁剪图的积累窗口 |
| 5000 μs | 原参考视频的常见物理采样间隔 |
| 25/30 FPS | 审核视频播放速率，仅影响慢放 |

转速用同一标记的两次观测时间差与角位移计算。全画面相邻事件间隔描述事件流活动，不应代替标记帧之间的 Δt。当前图像采样和 1 ms 积累仍可能造成簇中心的时间偏移；获得发球 RAW 后，可进一步在簇内使用原始事件时间精化中心与旋转估计。

## 使用与导出

```bash
bash scripts/pingpong.sh run event-timing \
  --raw data/camera_evidence/clock_endpoint_evidence_current/20261029_140835/trial_01_master.raw \
  --output output/timing_new
bash scripts/pingpong.sh spin --tracks output/latest/tracking/rotation_tracks_all.csv \
  --output output/spin_new
bash scripts/pingpong.sh run spin-check --spin-report output/spin_new/spin_report.json \
  --motion-csv output/latest/motion/ball_motion.csv \
  --timing-reports output/timing_new/timing_report.json --output output/spin_check_new
```

每个 `events_*.npz` 保存结构化 `events`（x/y/polarity/timestamp）、`delta_us`、`delta_valid` 和 `first_global_event_index`。第一事件没有前驱，`delta_valid=False`；后续批次首事件的 Δt 已跨批次计算。`adjacent_events_sample.csv` 包含前 5000 个事件及所有 SDK 批次边界；`delta_histogram.csv` 为完整间隔分布。

球速保持次要输出：本批 1801 个有效投影估计为 **1.54–6.98 m/s，中位数 3.49 m/s**，约为 5.5–25.1 km/h，未见极端数量级，未做数值截断。使用球径 40 mm 和局部弱透视假设，可用于工程参考；不等同于精确三维球速。
