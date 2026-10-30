# 工具索引

`src/pingpong_event_pipeline/` 存放共享逻辑，`tools/` 为历史兼容的命令入口。早期实验脚本之间和测试中存在跨文件导入；为避免破坏复现，尚未强行移动或删除这些脚本。旧研究结论不自动代表当前效果。

## 当前工作流

- `marker_dataset_launcher.py` / `marker_annotation_player.py`：发球组选择及播放标注；后者是核心 `annotation_ui` 模块的兼容入口。
- `batch_marker_crops.py`、`build_marker_label_dataset.py`：自动候选球心和事件裁剪。自动候选需人工审查。
- `review_robust_ball_tracks.py`：利用候选圆、轮廓证据和运动一致性生成旧/新轨迹对照；`export_refined_ball_crops.py` 在独立目录生成新裁剪，不迁移旧标注。共享关联位于 `src/pingpong_event_pipeline/ball_track_robust.py`。
- `review_event_refinement.py`：以粗轨迹为种子，逐事件平移并在 RAW 事件中细化球心；不覆盖标注或自动判定准确。共享实现位于 `src/pingpong_event_pipeline/ball_event_refine.py`。
- `audit_marker_dataset.py`、`export_marker_yolo.py`：核验、按发球分组导出标注；共享格式转换和导出逻辑位于 `src/pingpong_event_pipeline/annotations.py`、`yolo_export.py`。
- `export_single_serve_smoke.py`：单发球按时间划分的流程冒烟数据，**不是独立泛化测试**。
- `package_generalization_crops.py`：打包未标注候选用于观察，不可计算检测精度。
- `confirm_marker_drafts.py`：在用户确认标注完整后批量确认已有框；当前 UI 已支持画框自动确认。

## 采集与标定

- `dual_capture_isolated.py`、`dual_camera_tuner.py`、`live_ball_capture.py`、`live_measure.py`：采集、调参与诊断；对应常用快捷入口位于 `scripts/`。
- `manual_calibration_capture.py`、`calibrate_stereo.py`、`validate_stereo_calibration.py`：标定采集及验证；现有标定数据和使用边界请先人工检查。

## 历史研究

`analyze_*`、`audit_*`、`diagnose_*`、`estimate_*`、`probe_*`、`render_*`、`track_*` 等为研究和诊断工具。它们有交叉引用，某些依赖已经移入历史归档的数据；运行前应审查参数、依赖和结果，不要把早期试验结果作为当前测量事实。
