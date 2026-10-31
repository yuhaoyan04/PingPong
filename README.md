# PingPong

事件相机乒乓球项目：相机采集、球心追踪和运动补偿、分相机增强与去噪、人工标注、双相机分别训练 YOLO、标记簇追踪、离线球面旋转估计。

本工作区整合了 `yuhaoyan04/PingPong` 的 `3f08a83` 与现有 YOLO 工作区。**最新数据裁剪 → 两套已训练模型 → 连续三帧有效轨迹 → 转速候选与重投影审核，已经实际跑通。** Linux SDK 导入和同步测试 RAW 小窗口重放已验证；SDK 当前未发现实物相机。正式三维球速仍需要真实空间标定，物理转速精度还需要对应采集时间、球体几何与独立真值验证。

## 审核与运行

- [双视角时间与裁剪范围审核](output/latest_sync/review.html)：原双路全视野、严格同时间戳的共同裁剪；[同步审计说明](docs/CLIP_SYNC_AUDIT.md)。

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
