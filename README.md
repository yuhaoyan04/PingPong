# 事件相机乒乓球研究项目

本项目当前重点是两路 DVSLume 事件相机的发球数据、球面标记人工标注与模型实验。双相机时钟同步、自动球心轨迹和真实转速精度均未完全验证；生成的候选和可视化不能直接当作测量真值。

**仓库不包含数据或模型**：RAW、视频、图片、人工标注、标定结果、训练集、权重、官方 SDK 和实验产物都需自行准备。下文具体数据路径仅适用于已有本地实验工作区，克隆仓库后不会自动出现。

## 安装

使用 Python 3.9 或更新版本，在项目目录中创建环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
```

Linux/macOS 可使用 `source .venv/bin/activate`。`test` 包含测试与 PyQt 标注界面依赖，仅需要界面可安装 `.[gui]`。相机采集与 RAW 重放还需要另行安装相应系统和相机型号的 [动微官方 SDK](https://sdk.dvsense.com/zh/html/install_zh.html)，仓库不分发它；确认 `dvsense_driver` 可被 Python 导入。SDK 版本、相机、系统以及 GUI 功能需在目标机器上验证。

相机 SDK 1.3.0 安装、验证及官方资料入口见 [相机 SDK 说明](docs/hardware/CAMERA_SDK.md)；仓库内也包含保留原 MIT 许可证的官方 Python 示例快照。

Windows 启动脚本优先使用环境变量 `PINGPONG_PYTHON` 指定的解释器，否则尝试当前用户的 Anaconda，再回退 PATH 中的 `python`。配置和部分硬件脚本保留原实验使用的相机序列号与主从角色；实际使用前必须核对并修改，角色交换脚本不可直接套用于其他硬件。

## 快速入口

- 双击 `START_ALL_MARKER_REVIEW.cmd`，选择发球和相机进行标注。画框后自动保存；未能辨认的帧不要标为负样本。
- 九个原始回放片段：`data/output/isolated_capture/20261029_200352_785836/clips_checked/`。
- 对应两台相机的原始 RAW：同一采集目录下的 `master_DSP01F1FFF3E.raw` 和 `slave_DSP01F1FFF59.raw`。这次采集记录中 F3E 是 master、F59 是 slave；不代表未来接线和配置固定。
- 标注工作集和用户标注：`data/output/marker_annotation_all_v1/`；每组的 `binary/` 下同名 JSON 是人工标注，请勿覆盖或删除。
- 新的 RAW 事件候选轨迹和对比视频：`data/output/ball_pipeline_v1/`；相应的独立未标注裁剪集：`data/output/marker_annotation_event_v1/`。标注启动器可选择后者，使用前须人工核对，旧标注不会自动迁移。
- 双相机差异化增强标注工作集：`data/output/marker_annotation_camera_enhanced_v1/`，从启动器选择“双相机增强新工作集”。图像按 0.5 ms 步长、1 ms 窗口制作；只覆盖现有候选轨迹时段，并非完整原视频。回退帧在界面显示警告，旧标注不自动迁移。
- 标定采集及结果：`data/calibration/`。现有标定质量和适用范围仍需验证。
- 服务器训练及推理说明：`docs/training/`。

本实验机上的 `output/` 和 `calibration/` 是指向 `data/output/`、`data/calibration/` 的 Windows 目录联接，只用于兼容历史代码和已保存的原始路径；**不是第二份数据**。新克隆没有这些目录联接。不要删除联接指向的目录来“去重”。

## 目录结构

```text
src/pingpong_event_pipeline/  核心采集、事件、追踪、标注及测量代码
tools/                        CLI 与研究实验入口（历史脚本保留兼容）
scripts/                      常用相机、采集与标定启动脚本
tests/                        单元与工作流测试
config/                       配置
data/                         本地数据和产物（不提交）
docs/training/                服务器训练、推理提示词
docs/hardware/                硬件接线说明（不分发官方手册）
third_party/                  带许可证的官方 Python 示例快照
```

旧研究脚本仍可能引用本地数据、SDK 或旧目录，运行前请检查参数；工具用途及维护状态见 `docs/TOOL_INDEX.md`。本机另有不纳入仓库的历史资料，不代表远程仓库包含这些文件。

## 数据集导出

从项目目录执行以下命令，分组仅为示例；训练、验证、测试各组必须有已确认标注，且同一发球的两路相机不能跨集合：

```powershell
python tools\export_marker_yolo.py `
  --dataset data\output\marker_annotation_all_v1 --output data\output\marker_yolo_f3e `
  --train serve01_F3E --val serve02_F3E --test serve03_F3E
```

示例按 F3E 单独训练；F59 可用对应分组另行导出。未确认的图片不会自动当作负样本。空标注只有人工明确确认无标记后才可导出。当前自动球心可能跟错背景；导出前需要在视频中核对。重放时 SDK 曾报告时间戳异常，时间完整性与同步尚未彻底排除问题。

## 离线球体追踪候选

见 `docs/BALL_TRACKING_PIPELINE.md`。当前流程以已有粗轨迹为输入，使用 RAW 事件做运动补偿和局部圆周拟合，并对 18 组逐一生成诊断和回放；它仍是需要人工验证的候选，不是已经证明对所有新轨迹都鲁棒的自动检测器。自评分高不代表球心一定正确；低置信度或无有效支持的片段不能当作有效轨迹。

## 测试

```powershell
$env:PYTHONPATH='.sdk_1_3;src'
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest -q
```

硬件相关测试可能需要官方 SDK、设备或本地数据；请区分环境缺失与算法测试失败。
