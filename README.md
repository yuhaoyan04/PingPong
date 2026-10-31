# PingPong

事件相机乒乓球项目：相机采集、球心追踪和运动补偿、分相机增强与去噪、人工标注、双相机分别训练 YOLO、标记簇追踪、离线球面旋转估计。

本工作区整合了 `yuhaoyan04/PingPong` 的 `3f08a83` 与现有 YOLO 工作区。**最新数据裁剪 → 两套已训练模型 → 连续三帧有效轨迹 → 转速候选与重投影审核，已经实际跑通。** 已用实际发球双路 RAW 核对 5.475 亿个事件及旧参考画面的时间对应，并重建完整视野视频。两相机物理同步精度、球体几何与转速真值仍未完成独立标定；正式三维球速还需要空间标定。

## 审核与运行

- [原始结果与两组优化对照](output/latest_observations/review.html)：七段双视角三列视频，分别显示两路独立转速、系统总转速及标记局部方向；[方向修复说明](docs/SPIN_DIRECTION_REVIEW.md)、[定位与球几何算法](docs/OBSERVATION_REFINEMENT.md)。
- [全链路检查与迭代审核](output/latest_iteration/review.html)：训练配方对照、难例审核队列、固定几何下的转速/方向对照视频；[本轮结论与复现](docs/SYSTEM_ITERATION.md)。
- [标记中心轨迹平滑前后对照](output/latest_smoothing/review.html)：7 段双视角视频、固定窗口转速/方向、逐帧控制、原始与平滑坐标；[下载审核包](output/latest_smoothing/smoothing_review_bundle.zip)。
- [bbox 轨迹如何测转速、局部曲线平滑的公式与限制](docs/MARKER_TRAJECTORY_SMOOTHING.md)。新运行的转速、方向与运动可视化默认使用该平滑层，历史输出保留。

- [实际发球 RAW、轨迹与转速审核](output/latest_sync/review.html)：14 段新视频，完整双路视野、标记轨迹、四种半径下的转速与方向、逐帧控制；[下载审核包](output/latest_sync/raw_review_bundle.zip)。
- [纹理重复周期交叉检查](output/latest_sync/appearance/review.html)：不使用 YOLO 簇 ID 或球面拟合半径参数，保留重复分支歧义；[RAW 核验说明](docs/BALL_RAW_VALIDATION.md)。
- [RAW 球边界上的四种半径对照](output/latest_sync/raw_radii/review.html)：同一原图、同一球心，直接审核假设半径与对应转速。
- [此前导出裁剪范围审核](output/sync_audit_v8/review.html)；[同步审计说明](docs/CLIP_SYNC_AUDIT.md)。

- [逐帧转速、方向与四种球半径对照](output/latest_radius/review.html)：同步数值、方向箭头、逐帧暂停与单格放大。
- [半径对照视频审核包](output/latest_radius/radius_review_bundle.zip)；[显示规则与半径说明](docs/RADIUS_VIDEO_REVIEW.md)。

- [逐次发球双视角联合审核](output/latest_serve/review.html)：共享转速模型、同步视频、逐组失败原因与单视角回退。
- [双视角审核下载包](output/latest_serve/serve_review_bundle.zip)；[零转速与联合建模说明](docs/SERVE_TWO_VIEW_SPIN.md)。

- [转速、方向与逐事件时间优先审核](output/latest_spin/acceptance/review.html)：文献对照、有向轴与方向稳定性、全量事件间隔。
- [转速与逐事件时间审核包](output/latest_spin/spin_review_bundle.zip)：解压后打开 `acceptance/review.html`，含全量事件间隔 NPZ。
- [转速与时间验收说明](docs/SPIN_TIMING_ACCEPTANCE.md)。

- [最新完整运动审核页](output/latest/motion/review.html)：标记轨迹、球心路径、像素球速、条件横向速度与转速候选，可按时间查看。
- [离线审核下载包](output/latest/review_bundle.zip)：解压后打开 `motion/review.html`。
- [逐窗口结果](output/latest/spin/spin_windows.csv)、[分组汇总](output/latest/spin/spin_by_group.csv)、[完整报告](output/latest/spin/spin_report.json)。
- [连续有效轨迹](output/latest/tracking/rotation_tracks_all.csv)；每相机每组的 `comparison_slow.mp4` 与 `overlays/` 可审核检测和追踪。
- [SDK 与运动分析新流程](docs/PIPELINE_MOTION.md)
- [统一流程说明](docs/UNIFIED_WORKFLOW.md)、[转速算法与优化设计](docs/OFFLINE_MARKER_SPIN.md)。

```bash
cd /data2/songxinshuai/yanyh/PingPong
bash scripts/pingpong.sh doctor
# 仅使用现有三帧有效轨迹，不重复推理；输出目录须不存在
bash scripts/pingpong.sh spin --output output/spin_new
# 重新推理、追踪、筛选、估计转速候选和生成审核页
bash scripts/pingpong.sh pipeline --camera all --device 1 --output output/pipeline_new
# 原有各环节参数通过统一入口传递
bash scripts/pingpong.sh run infer --help
bash scripts/pingpong.sh run ball-track --help
```

Windows 或其他解释器可以运行 `python pingpong.py ...`，参数相同。Linux 可用 `PINGPONG_PYTHON` 指定解释器。GUI 标注需要图形会话；远程审核可下载 HTML 及同目录 PNG、CSV、JSON。

## 目录与资产

| 路径 | 内容 |
| --- | --- |
| `src/pingpong_event_pipeline/` | 采集、事件、追踪、增强、标注、原有旋转求解器与新轨迹转速适配器 |
| `tools/`、`tools/yolo/` | 原有相机/数据工具与汇入的 YOLO 训练、推理、追踪代码 |
| `config/`、`configs/inference_tracking.json` | 相机、增强、转速与追踪参数 |
| `data/server_marker_two_camera_train_v1/` | 当前唯一最新数据集，保留源包校验和 |
| `experiments/two_camera_v1/` | 分相机训练记录、各自 `train/weights/last.pt` |
| `experiments/tracking_min3_v1/` | 原有连续三帧结果，保留作此次集成输入 |
| `output/latest/` | 完整重新推理、轨迹、球速/转速候选与视频结果 |
| `output/integration_assets.json`、`output/integration_validation.json` | 来源、模型哈希与验证记录 |
| `tests/`、`tools/yolo/test_*.py` | 原项目与新增算法/边界测试 |

数据、权重、环境和输出不提交 Git。原 `Yolo-PointTrack` 保留；本项目数据、模型、历史结果和 `.venv` 都是独立副本。环境沿用服务器 Anaconda 的 PyTorch/CUDA，没有升级驱动或 CUDA。新机器仍需准备环境和数据。

## 环境与测试

当前 Python 3.12、PyTorch 2.6.0+cu124、Ultralytics 8.4.192、NumPy 1.26.4、Numba 0.60.0。训练依赖锁定见 `requirements-yolo.txt`，基础依赖见 `pyproject.toml`。新建环境先安装匹配硬件的 PyTorch，再安装项目依赖，避免覆盖工作的 CUDA 环境。

```bash
PYTHONPATH=src:tools/yolo QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests tools/yolo/test_*.py
```

仓库已包含 Windows SDK 安装包、官方示例和相机手册，详情见 [SDK 说明](docs/hardware/CAMERA_SDK.md)。Linux/Python 3.12 使用匹配的官方 wheel，不能安装 Windows cp39 包。实际采集仍需核对序列号和主从接线。历史 Windows 路径见 [上游原说明](docs/UPSTREAM_WORKFLOW.md) 与 [工具索引](docs/TOOL_INDEX.md)，历史路径不代表相应采集数据已同步；本机资产状态以 `doctor` 和 SDK 运行报告为准。

本轮完整运动报告见 `output/` 中的新 pipeline 运行。源代码已合入上游最新 SDK 与文档更新，数据、权重和本地证据仍不提交远程。
