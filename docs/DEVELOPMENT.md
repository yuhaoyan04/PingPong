# 开发与验证

## 环境

日常修改在新分支完成。CPU 测试环境与 CUDA 训练环境分别维护：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m compileall -q src tools pingpong.py
```

`pyproject.toml` 配置了测试目录和源码路径，安装项目后无需手动设置 `PYTHONPATH`。`.github/workflows/tests.yml` 在 main 更新及拉取请求时，以 Python 3.10、3.12 检查安装、依赖一致性、命令行、语法和全部 CPU 测试。硬件采集、模型训练与真实转速准确率不由这些测试替代。

训练沿用 `requirements-yolo.txt` 与本机兼容的 PyTorch/CUDA，不将 CPU 测试依赖直接覆盖到正在工作的训练环境。

## 输入和数值

- 每个相机序列的图片、源帧号、事件时间必须与 manifest 一致。不能用视频慢放 FPS 计算运动。
- 预测保持不算观测，轨迹导出至少连续三帧；旋转求解另检查观测数、时间跨度、球几何和残差。
- 单相机角速度与系统共享旋转保持分开。未知值是 N/A，有近零旋转证据才显示近零。
- 改变裁剪球心/半径时重新做事件补偿，不只缩放最终 RPM；新几何下不复制旧图像标注。
- 关键追踪输入哈希和时间校验使用显式异常，`python -O` 不会关闭这些检查。

## 审核结果缓存

```bash
python pingpong.py run review-observations \
  --source output/continuity_new --output output/review_new

# 只审核指定发球，可用于快速验证显示改动
python pingpong.py run review-observations \
  --source output/continuity_new --output output/review_probe --serves serve05

# 强制重新编码
python pingpong.py run review-observations \
  --source output/continuity_new --output output/review_new --force-render
```

每段视频旁的 `*.cache.json` 记录输入指纹、视频 SHA256、尺寸、FPS 和全帧解码检查。输入指纹包含本段轨迹与拟合、图像像素、manifest、metadata、显示标签、渲染代码、字体和关键库版本。只有这些输入以及视频字节都未改变时才复用。单纯存在同名 MP4 或帧数相同不构成复用依据。

修改图像、观测、拟合、配置所影响的结果、字体或显示代码后，对应缓存会失效。`--serves` 用于部分审核，不会更新 `output/latest_observations`；发布最新入口前应生成并检查全部目标发球。`video_validation.json` 标记 `cache_hit`，区分重新编码与复用已验证字节。

源 profiles 可采用相对 `variants.json` 所在目录的路径。三个源旋转报告的观测哈希会在更新输出前统一检查；轨迹变了而报告未重算时明确失败，不能继续展示旧数值。旧目录没有缓存记录的视频会重新编码一次。

新 MP4 在临时目录编码、解码验证后才替换正式文件；编码失败、帧数不符时保留旧视频和缓存。JSON、HTML 与 ZIP 同样采用原子替换，避免把中断写入的文件当成完整结果。哈希采用分块读取，处理大型 RAW 时不会一次把整个文件载入内存。不要同时向同一个输出目录运行多个审核任务。

## 提交前检查

```bash
git diff --check
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m compileall -q src tools pingpong.py
```

算法修改除合成测试外，应使用冻结检测重放或真实 RAW 对照，检查源时间、轨迹连续性、单视角和联合拟合的失败原因。可用窗口或观测数量增加不直接等于精度提升。输出目录保留代码提交、参数、输入哈希和验证记录。

数据、权重、RAW、环境、视频及本地凭据不提交到仓库。SDK 资料作为已纳入版本控制的项目资产保留。推送前先获取上游更新，再正常合并或快进，不强制覆盖远端历史。
