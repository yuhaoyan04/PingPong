# 实际发球 RAW、时间与纹理重复周期

审核：`output/latest_sync/review.html`，实际目录 `output/raw_sync_v9/`；下载 `raw_review_bundle.zip` 后可离线打开 HTML。包含 7 次发球的 14 段视频：每次一段 RAW/旧参考对照，一段 RAW 全视野、连续标记簇轨迹及四种半径下转速/方向候选。视频采样 5 ms 或 0.5 ms，播放 30/25 FPS；算法计算使用整数事件时间。

## 实际文件与逐事件核验

输入为 `data/master_DSP01F1FFF3E.raw`、`data/slave_DSP01F1FFF59.raw`。序列号从文件头核对，不套用之后同步测试的主从角色。源 SHA256、尺寸、输入模型/轨迹/manifest 哈希保存在报告与 `input_assets.json`。

| 相机 | CD 事件数 | 首事件 μs | 末事件 μs | 负间隔 |
| --- | ---: | ---: | ---: | ---: |
| F3E | 280,819,184 | 643,884,000 | 671,343,746 | 0 |
| F59 | 266,703,249 | 643,884,000 | 671,935,295 | 0 |

完整 SDK 顺序事件与独立 EVT3 字节解码的数量、全部 x/y/极性/时间顺序 SHA256 一致。事件未排序、去重或人为替换时间；相邻间隔包含 SDK 批次边界。此次记录的相邻间隔均为 0 或 1 μs；同一个微秒量化格内的多个事件不证明物理同时。微秒量化也不是微秒物理精度的校准证明。

SDK 的 EOF 标志曾使 F3E 末尾 19,184 个事件漏读；已改为读取到空批次，并再次与独立解码核验。首次失败保留在 `timing_F3E_initial_eof/`，正式报告为 `timing_F3E/`。建索引或 seek 期间出现的 SDK 时间警告没有被当作可直接删事件的依据；完整原始 CD 序列中未发现倒退。另修复顺序时间批次复用 SDK 缓冲区的问题，保留的旧批次现在持有自己的副本。

## 导出映射与同步范围

每次发球完整参考范围内，两路独立解码事件均与 SDK 顺序读取及提前 1 ms 的 seek 逐事件一致。两路都用同一绝对 `[t−1000,t)` 窗口重建。890 个旧双路参考帧的占用前景像素，在排除顶部烧录字样并统一阈值后与 RAW 重建一致；164 个抽样在 ±10 ms、1 ms 步长搜索中的最佳偏移为 0。这验证各自 RAW 到旧参考的时间映射，不是两个物理时钟之间的独立校准。

两份文件未记录外部触发事件。500 μs 全局活动直方图的互相关峰低且在时段/极性之间不稳定，最高约 0.212，未接受任何时移。文件日期、首事件或球进入视场的时刻都没有被用作物理时钟锚点。报告 `export_mapping_verified=true`、`physical_camera_synchronization_verified=false`；RAW 已上传，不再等待上传。新的完整视野可以人工检查某一侧没有裁剪时是否仍有球。

## 纹理重复周期交叉检查

`appearance/review.html` 直接比较归一化球内部纹理：256 px 图像先以 σ=2 px 模糊、缩至 48 px，取名义球半径 0.78 倍内的像素，逐帧去均值及单位范数，再计算两帧相关矩阵与同延迟平均相关。它不使用 YOLO 簇 ID，也不使用 SO(3) 的球面半径参数，但仍依赖已有球心/尺度归一化。

保留至少 16 对图像；搜索 3–80 ms 内可观察的延迟。双侧峰相关至少 0.2、延迟相差不超过 `max(1 ms, 8%)` 才列出共同分支。两侧各自全段分析，裁剪时间范围不同，因此只比较重复延迟，不把这项测试当作同步证据。红线仅显示既有拟合，不参与选峰。

| 发球 | 双侧相近纹理重复延迟 | 1/T，次/秒 |
| --- | ---: | ---: |
| 02 | 28.75 ms | 34.78 |
| 03 | 24.25 ms | 41.24 |
| 04 | 23.00 ms | 43.48 |
| 05 | 未找到共同分支 | N/A |
| 07 | 45.75 ms | 21.86 |
| 08 | 25.75 ms | 38.83 |
| 09 | 7.25 / 26.50 ms | 137.93 / 37.74 |

02/03/04/07 的纹理重复分支接近既有联合转速候选，说明高值并非仅由球面半径拟合产生。09 同时存在明显短周期与较长周期分支，不能自动选择其中一个。多个相似标记、半周/多周重复、归一化抖动和噪声均可能形成假峰；这些是纹理重复频率，不能直接认定物理转速。重复示例每行独立选起点，时间印在图上，不是双目同时帧。没有重复证据不等于不转球。

本轮不修改已有转速来匹配预期。方向仍为相机局部坐标下有向轴与可见前表面运动箭头，世界坐标上下旋仍需空间标定。后续应检查唯一标记身份、球轮廓/球心残差，以及用已知转速的独立样本校准；不得通过人为放大半径来让数值看起来合理。

`raw_radii/review.html` 还提供 7 张 RAW 球边界对照图：两行是同一个绝对时间的相机画面，每行四列固定原始像素和球心，仅叠加 ×1、×1.1、×1.2、×1.5 的半径假设及该时间对应转速。可直接判断较低转速是否以明显偏离球边界的半径为代价；这些假设圆不是新的测量结果。

## 全量相邻事件时间的紧凑编码

`timing_F3E/event_clock_runs.npz` 和 F59 同名文件无损保留所有事件时间，分别约 4.7 MB 和 3.2 MB。`counts[k]` 表示连续多少个原始事件拥有时间 `first_timestamp_us + k`。因为完整顺序已核实单调，它保留每一个 0/1 μs 间隔及批次边界。首次事件的前一间隔不存在。序列 SHA256 对应完整像素/极性/时间事件记录，不只是时间计数。

```python
import numpy as np
z = np.load('timing_F3E/event_clock_runs.npz')
counts = z['counts']
# 原事件序号 i 对应的微秒格；只创建约 2800 万项索引，不展开 2.8 亿时间戳。
last_indices = np.cumsum(counts, dtype=np.int64)
i = 12345
k = int(np.searchsorted(last_indices, i, side='right'))
t_us = int(z['first_timestamp_us']) + k
previous_k = int(np.searchsorted(last_indices, i - 1, side='right')) if i else None
delta_us = k - previous_k if i else None
```

## 复现

各路径使用新目录，缓存保存独立顺序解码事件，约 1.1 GB；原始 RAW 不改写。时间与像素审核不会排序修补异常。

```bash
cd /data2/songxinshuai/yanyh/PingPong
bash scripts/pingpong.sh run event-timing --raw data/master_DSP01F1FFF3E.raw --output output/new_raw/timing_F3E --no-event-export
bash scripts/pingpong.sh run event-timing --raw data/slave_DSP01F1FFF59.raw --output output/new_raw/timing_F59 --no-event-export
bash scripts/pingpong.sh run ball-raw-audit --raw data/master_DSP01F1FFF3E.raw --camera F3E --output output/new_raw --cache /tmp/pingpong_new_raw_cache
bash scripts/pingpong.sh run ball-raw-audit --raw data/slave_DSP01F1FFF59.raw --camera F59 --output output/new_raw --cache /tmp/pingpong_new_raw_cache
bash scripts/pingpong.sh run event-clock-runs --raw data/master_DSP01F1FFF3E.raw --timing-report output/new_raw/timing_F3E/timing_report.json --output output/new_raw/timing_F3E/event_clock_runs.npz
bash scripts/pingpong.sh run event-clock-runs --raw data/slave_DSP01F1FFF59.raw --timing-report output/new_raw/timing_F59/timing_report.json --output output/new_raw/timing_F59/event_clock_runs.npz
bash scripts/pingpong.sh run ball-raw-video --output output/new_raw --cache /tmp/pingpong_new_raw_cache
bash scripts/pingpong.sh run appearance-period --output output/new_raw/appearance
bash scripts/pingpong.sh run raw-radius-review --output output/new_raw/raw_radii --cache /tmp/pingpong_new_raw_cache
bash scripts/pingpong.sh run ball-raw-package --output output/new_raw
```
