# 离线球体追踪候选流程

本次使用九个发球、两台相机共 18 组已有数据。当前流程依赖 `trajectory_refinement_v2` 中已经生成的粗候选，并不是从任意新 RAW 自动发现全部球轨迹的完整检测系统。

本文档记录本机实验；示例中的数据、输出、人工标注、SDK 和模型均不在 Git 仓库中。新环境需自行采集或提供相应输入。

1. 对粗球心进行稳健平滑并估计运动速度。
2. 对 1 ms RAW 事件窗口按平移速度补偿，在候选位置附近搜索圆周支持；该步骤没有补偿球的旋转。
3. 对有局部事件支持的球心做离线平滑；跨缺口不平滑。无支持时保留回退结果用于诊断，不当作有效追踪。
4. 每组保存对比视频、逐时刻轨迹和诊断。事件圆周支持仅是启发式分数，不是实际球心准确率；轨迹需要人工核验。
5. 单独导出连续有支持区域的二值与极性裁剪图；不跨支持缺口生成裁剪，且不复制原裁剪图上的人工标注。新裁剪采用新坐标系，旧标注不能直接复用。

## 当前结果

- `data/output/ball_pipeline_v1/summary.json`：18 组汇总和诊断标记；每组 `comparison_raw_1ms.mp4`、`raw_refined_path.npz`、`quality.json`。
- `data/output/marker_annotation_event_v1/`：17 组、1824 张独立的候选裁剪；第 5 球 F3E 没有足够长的连续支持，未导出。其他存在低支持率等标记的组同样需要重点复核。
- 对比视频绿色为平滑粗候选，紫色为局部 RAW 修正；回退状态会标识。画面仍需与真实球体逐帧对照，颜色不代表真值。
- 标注界面视频时间粒度与裁剪图不完全相同，原始视频大约 5 ms 一帧，裁剪图以 0.5 ms 步长生成，展示时有最近帧时间差。
- SDK 重放曾报告时间戳异常，双机同步仍未证实。这些候选不能用于宣称准确的三维速度或转速。

## 复现

从项目根目录运行；输出目录必须不存在，避免覆盖已有成果：

```powershell
$env:PYTHONPATH='.sdk_1_3;src'
python tools\run_ball_pipeline.py `
  --groups data\output\marker_annotation_all_v1 `
  --coarse data\output\trajectory_refinement_v2 `
  --output data\output\ball_pipeline_new
python tools\export_supported_ball_crops.py `
  --pipeline data\output\ball_pipeline_new `
  --originals data\output\marker_annotation_all_v1 `
  --output data\output\marker_annotation_event_new
```

## 双相机增强标注工作集

`data/output/marker_annotation_camera_enhanced_v1/` 是 18 组独立标注工作集。统一配置 `config/marker_enhancement_v1.json`：F3E 轻度邻域过滤（3 像素半径、至少 2 个邻居），F59 中档多尺度对比过滤（阈值 0.035）。双击 `START_ALL_MARKER_REVIEW.cmd`，选择“双相机增强新工作集”，再选发球和相机。框选后沿用自动保存；切片坐标与旧工作集不同，不复制已有标注。不可辨认时不要确认无标记。

每组 `binary/` 是待标注增强图，`binary_unfiltered/` 是同坐标未去噪图，`polarity_reference/` 是未过滤的蓝/黄事件参考；`manifest.csv` 记录时间和球心。裁剪半径是候选半径的 1.2 倍，只是边距并非真实半径。每 0.5 ms 导出一次、每张积累 1 ms 事件；原视频以 5 ms 间隔显示最近帧，界面会显示时间偏差。

为了不丢掉已有候选中的部分图像，本工作集也导出了低置信回退帧，逐帧在 `track_support` 和标注界面标注；回退位置并非可靠检测。即使标记为圆周支持，也只是启发式并需检查球心和半径。跨时间缺口不插值；候选轨迹没有覆盖的时段没有生成球体裁剪，不能说已切出原视频每个时刻。SDK 的时间戳异常和双机同步仍未解决。

重建须指定一个不存在的新输出目录，避免覆盖人工标注：

```powershell
$env:PYTHONPATH='.sdk_1_3;src'
python tools\export_supported_ball_crops.py `
  --pipeline data\output\ball_pipeline_adaptive_v2 --originals data\output\marker_annotation_all_v1 `
  --output data\output\marker_enhanced_new --crop-scale 1.2 `
  --camera-profiles config\marker_enhancement_v1.json --include-fallback
```

## 验证缺口

需要按发球和相机抽取分布在起始、快速运动、拖尾、遮挡及视野边缘的样本，人工标记真实球心和半径；按未参与调参的发球统计像素误差、覆盖率、断轨、误检与速度分层指标。没有这些真值时，不能以圆周支持率或者与旧算法的一致性证明泛化能力。对于缺失粗候选或球像被拖尾淹没的情况，还需要独立的重检测和恢复机制。

## 裁剪余量、去噪和延长片段试验

- `data/output/ball_crop_denoise_review_v1/<组>/crop_comparison.mp4` 四栏依次为旧球区、半径乘 1.35 的保守裁剪、邻域去噪、红色显示被删事件；所有结果仍使用旧候选中心。扩大裁剪范围不是新估计的真实球半径，且可能纳入背景。第 5 球 F3E 的轨迹几乎全是未被验证的回退值，不能误作可靠跟踪。
- 去噪以两极性合并后的占据像素为输入；在 256×256 裁剪图中，保留 3 像素邻域内至少存在另外 2 个占据像素的点。它不按球轮廓删除像素，也不填充或合并簇；真实稀疏标记可能被误删，连续边缘噪声则不一定能去除。
- `data/output/marker_annotation_padded_denoised_v1` 是独立待复核工作集：`binary` 是去噪图，`binary_unfiltered` 是相同坐标系的未去噪图，`polarity_reference` 保留原极性图；它不会复制旧标注。只有连续得到支持的轨迹段被导出，第 5 球 F3E 不在其中。
- `data/output/extended_clip_reviews_v1` 依据旧视频的起始时刻和帧数，从原始 RAW 重新渲染了前 200 ms、后 300 ms 的候选延长视频。新增时间范围不意味着球一定可见或已被追踪；视频每 5 ms 取一个 1 ms 事件窗口，仅供人工查看。RAW 重放仍有时间戳异常警告，两相机同步未验证。

可通过下列参数重新生成实验性工作集；输出目录不可已存在：

```powershell
$env:PYTHONPATH='.sdk_1_3;src'
python tools\export_supported_ball_crops.py `
  --pipeline data\output\ball_pipeline_v1 --originals data\output\marker_annotation_all_v1 `
  --output data\output\padded_trial --crop-scale 1.35 --denoise `
  --neighbor-radius 3 --min-neighbors 2
```

## 自适应半径与 F59 时序过滤试验

`ball_pipeline_adaptive_v2` 是独立离线实验，不会替换原来的追踪或标注。旧版本的 ±4 px 圆周搜索是人为设计的抗干扰约束，不是相机限制。新版本按粗半径扩大搜索范围，上限 45 px，并在有连续支持时限制相邻时刻的半径变化；但它仍依赖旧粗轨迹的时段与中心，只在局部搜索，不具备完整画面的实时重检。圆周分数高也可能是球体边缘、拖尾或噪声，不能当作真值。

`adaptive_full_trajectory_review_v2/<组>/full_trajectory_and_denoise.mp4` 将全画面 RAW 事件和局部图同时显示：绿圈为旧圆（底部显示其支持/回退），紫圈及轨迹为新算法有支持的结果，橙圈表示新算法回退，白圈表示额外 1.2 倍裁剪范围，并非球体实测半径。右侧依次为裁剪原图、空间邻域去噪、空间与上一窗口的邻域活动联合过滤。时序结果仅在相邻两帧均有支持时显示；该方法可能删除新出现或快速旋转的真实标记，也不能可靠去除持续存在的噪声。

每帧对应 1 ms 事件窗口、1 ms 采样步长，按 25 fps 慢放；原始事件未做旋转补偿。新算法诊断仍需人工复核，F3E 第 5 球粗候选少且支持片段短。较低跳动或较高支持分数不证明圆心、半径准确，也不能证明 F59 的标记点已被成功分离。

## F59 多尺度局部对比度过滤试验

`data/output/f59_contrast_denoise_review_v1` 包含九段 F59 的全画面与局部对照：原图、旧邻域过滤、中档局部对比度、强档局部对比度、强档被删除像素的红色标记。所有视频只覆盖当前自适应候选轨迹的时间范围，并非原始发球完整时段。`pixel_counts.csv` 是不同输出的保留像素数，不是噪声量或标记点准确率。

局部对比度先将两极性合并为二值占据图，计算小尺度高斯密度减大尺度密度（sigma 3、12，单位为 256×256 裁剪像素），中档阈值 0.035、强档 0.080，最后只保留满足阈值的原始占据像素；没有重建新的事件，也不做旋转补偿。高亮的球边缘和拖尾可能被当成局部高密度结构保留；真实但稀疏、低对比度的标记可能被删。强档不应未经人工核验用于训练集或测量。

```powershell
$env:PYTHONPATH='.sdk_1_3;src'
python tools\review_f59_contrast_filter.py `
  --groups data\output\marker_annotation_all_v1 --pipeline data\output\ball_pipeline_adaptive_v2 `
  --output data\output\f59_contrast_trial --moderate 0.035 --strong 0.08
```
