# 统一工作流

所有命令在项目根目录运行。`bash scripts/pingpong.sh run <环节> ...` 保留各工具参数，共用同一 Python 环境与源码导入路径。输出采用新目录，不覆盖标注和已完成训练。

| 环节 | 命令 | 输入 → 输出 |
| --- | --- | --- |
| 环境检查 | `doctor` | SDK、环境、数据、模型哈希、RAW 和标定状态 |
| 双机采集 | `run capture` | 实物相机、实际主从角色 → 两路 RAW、配置回读和记录 |
| 标定 | `run calibration-capture` / `run calibrate` / `run validate-calibration` | 实物采集 → 标定和验证 |
| 新 RAW 球心检测 | `run ball-track` | RAW、时间段、搜索 ROI、半径范围 → `monocular_path.npz` 和视频 |
| 新组接口 | `run prepare-group` | 检测 NPZ、RAW、参考视频和物理时间 → `groups/<组>/source.json`、`coarse/<组>/candidate_tracks.npz` |
| 球体精化 | `run ball-refine` | 组与粗候选 → `raw_refined_path.npz`、圆周支持诊断和视频 |
| 运动补偿、裁剪、增强/去噪 | `run enhance` | 精化轨迹、RAW、分相机配置 → 二值、未过滤、极性图与坐标/时间清单 |
| 人工标注 | `run annotate` | 组、参考视频、物理时间映射 → 同名 Labelme JSON、reviewed 标志 |
| YOLO 导出 | `run export-yolo` | 确认标注、按发球分组 → 各相机图片/标签/YAML |
| 当前服务器包 | `run package` | 当前九个发球工作集 → 双相机数据、未标注推理集、校验和 |
| 当前数据审计 | `run audit` | 最新数据包 → 训练副本与来源/像素/标签审计 |
| 分相机训练 | `run train` | 单通道图、官方初始化 → 各自固定 epoch 最终 EMA `last.pt` |
| 标记追踪和三帧筛选 | `run infer` | 匹配相机的模型与序列 → 视频、连续三帧以上轨迹 |
| 离线转速 | `spin` | 连续轨迹、源 manifest → 候选角速度、RPM、残差、审核页 |
| 当前数据完整离线闭环 | `offline` | 最新裁剪与模型 → 推理、追踪、筛选、转速候选、审核页 |

`legacy-nine-serve-candidates` 保留上游九段视频的硬编码时码；新 RAW 使用 `ball-track` 和 `prepare-group`。自动球心仍需人工复核，不保证发现任意轨迹。

## 当前版本复用

只使用 `data/server_marker_two_camera_train_v1`：F3E 213 张、246 框，F59 165 张、237 框。serve01/06 全用于训练；另 7 次发球合计 1,946 张用于未标注定性推理。没有独立带标注验证/测试集，不发布检测或转速准确率。

权重为 `experiments/two_camera_v1/<相机>/train/weights/last.pt`，不是验证集选择的 best。原数据不变，缓存与绝对路径 YAML 位于 `experiments/.../working_data`。集成仅更新工作副本路径，原训练记录中的旧路径作为来源保留。

```bash
bash scripts/pingpong.sh offline --camera all --device 1 --output output/offline_new
bash scripts/pingpong.sh run infer --camera F3E \
  --source data/server_marker_two_camera_train_v1/unlabeled_inference/F3E/serve02_F3E \
  --device 1 --output output/serve02_f3e_new
bash scripts/pingpong.sh spin --tracks output/serve02_f3e_new/rotation_tracks_all.csv \
  --output output/serve02_f3e_spin_new
```

`offline --replay-cache` 用于冻结检测的后处理比较，会注明缓存来源；此次 `output/latest` 是 GPU 1 的实际重新推理。通道检查挂在实际预测后端，支持模型融合后连续处理多组序列。

## 新采集到标注

先检查 SDK，在有相机和图形会话的机器使用 `run capture --help`，显式指定实际 master/slave。序列号不等于固定角色。采集保存配置回读，保留测转速所需纹理。Linux SDK 已安装并通过同步测试 RAW 小窗口重放，但 SDK 当前未发现相机，真实发球 RAW 与真实空间标定仍缺失；更新状态与运动报告入口见 `PIPELINE_MOTION.md`。

```bash
# 依据新 RAW 决定时间段、搜索 ROI 和半径，不能套历史九段时码
bash scripts/pingpong.sh run ball-track --help
bash scripts/pingpong.sh run prepare-group --help
```

`ball-track` 的 `monocular_path.npz` 接入 `prepare-group`，后者只将正分数真实检测作为精化种子，不用插值/预测冒充观测。为各相机/发球准备组后运行：

```bash
bash scripts/pingpong.sh run ball-refine \
  --groups output/prepared_new/groups --coarse output/prepared_new/coarse \
  --output output/ball_refined_new --adaptive-radius
bash scripts/pingpong.sh run enhance \
  --pipeline output/ball_refined_new --originals output/prepared_new/groups \
  --output output/annotation_new --crop-scale 1.2 \
  --camera-profiles config/marker_enhancement_v1.json
```

默认导出有支持的连续球体片段；显式 `--include-fallback` 的回退帧需重点复核，转速仍排除它们。新裁剪坐标改变后不复制旧标注。F3E 邻域 radius=3/min_neighbors=2；F59 多尺度 sigma=3/12、threshold=0.035。对照 `binary_unfiltered` 和极性图检查有没有删掉真实标记。

`run annotate --group ... --video ... --video-first-us ... --video-step-us ...` 打开标注界面。看不清的帧不能确认无标记。参考视频第一帧与步长必须用物理时间，不用 30 FPS 回放速度代替。

`export-yolo` 支持整次发球 train/val/test 划分，防止两相机同次发球跨集合。当前 `package` 对应九次发球、serve01/06 训练方案；扩展发球和集合必须调整打包与审计规则，不通过改名绕过检查。

## 复现当前训练

现有模型已完成，整合未重训。复现使用新实验目录：

```bash
export PINGPONG_EXPERIMENT="$PWD/experiments/two_camera_reproduce_new"
bash scripts/pingpong.sh run audit
bash scripts/pingpong.sh run train --camera F3E --device 1 --epochs 150
bash scripts/pingpong.sh run train --camera F59 --device 1 --epochs 150
unset PINGPONG_EXPERIMENT
```

初始化权重 `weights/yolo11n.pt`；依赖 `requirements-yolo.txt`。保持单通道二值图，关闭会改变事件几何/颜色的随机增强；包括末轮在内禁用验证与 best 选择。YAML val 训练别名仅为框架兼容，不是独立评估。新增数据集可用 `PINGPONG_DATASET` 配置路径，但必须符合当前包格式和审计规则。

## 转速输入契约

每段至少三帧，同一相机/组/ID，原始图索引与微秒时间严格连续；只接受检测或有证据的光流。缺帧、超时、纯预测时断段，不拼接分散的三个点。求解再次核对 manifest 的图片、时间和支持状态。

无 manifest 的序列可按假设帧率预览追踪，但转速入口拒绝缺失源清单的输入。自定义包需指定 `spin --dataset`；`offline` 当前连接最新包，不能将猜测的帧率当已验证时钟。
