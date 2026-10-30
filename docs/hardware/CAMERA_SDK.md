# 动微相机 SDK 与资料

## 本次 Linux 集成验证（2026-10-31）

已合入 GitHub `3f08a83`。本服务器是 Ubuntu 20.04 / CPython 3.12，仓库的 Windows cp39 wheel 不适用；已从官方 PyPI 安装 `dvsense-driver==1.3.0` 的 `cp312-manylinux_2_28_x86_64` wheel。安装文件及 SHA-256 记录在本地 `data/sdk/`，不覆盖 NumPy、PyTorch 或 CUDA。

```bash
bash scripts/install_camera_sdk.sh
bash scripts/pingpong.sh run sdk-check
bash scripts/pingpong.sh run sdk-audit --help
```

本次实际验证结果见 `output/sdk_validation_v2/report.json`：Python SDK 和三个核心模块导入成功；一份同步测试 RAW 的 10 ms 小窗口成功解码，得到 1,783 个坐标合法、时间单调的事件。仅验证了这个解码窗口，不代表整份文件时钟或训练采集时间已通过验证。SDK 枚举未发现相机，没有打开设备或开始采集。

厂商当前的 [原生 SDK 系统支持说明](https://sdk.dvsense.com/zh/html/install_zh.html) 列出 Ubuntu 22.04/24.04，1.3.0 不再正式支持 20.04。这里记录的是本机 Python wheel 的实测导入和小窗口重放结果，不能推广成实机驱动兼容保证；无需为本次离线分析改动系统 apt、驱动或 CUDA。

本地上传的近期同步记录已复制到 `data/camera_evidence/`；其中的时钟测试 RAW 不是乒乓球训练数据，不进入训练集。自定义验证工具归入 `tools/camera_diagnostics/`，官方示例仍单独保存在 `third_party/dvsense_driver_python_sample/`。历史 PASS 记录不能直接证明另一段采集的时钟和双目视觉时间一致。

采集与手动标定入口的默认角色已统一为当前配置的 F59=MASTER、F3E=SLAVE，从机先启动；实际接线变动时仍需显式设置并重新验收。这不会改变旧录制或旧标定文件中相机的顺序。

真实的发球 RAW 和真实空间标定文件目前仍未提供。现有样本可以进行 YOLO、追踪、像素运动与转速候选分析；有标定文件之后，统一入口可计算基于三角化的三维速度候选。

本项目此前实际使用 `dvsense-driver 1.3.0`、Windows x64、CPython 3.9 的 Python 驱动。官方包还有其他 Python 和平台的构建；使用前核对平台、Python 版本、相机型号和 SDK 兼容性。本仓库不包含本地虚拟环境，也没有把相机数据、序列号对应的采集文件或标定结果打包进来。

## 安装与检查

先按照根目录 README 创建并激活 Python 环境，再从官方 Python 包仓库安装本项目实验使用的版本：

```powershell
python -m pip install -r requirements-camera.txt
python tools/check_camera_sdk.py
```

Windows 用户也可以在激活环境后运行 `./scripts/INSTALL_CAMERA_SDK.ps1`；如需指定解释器，可传 `-Python <python路径>`。导入检查不会连接相机，也不能证明双机同步正常。安装驱动、上位机或解决 USB 设备权限时应参考厂商的系统安装说明；Linux 系统级库及支持版本也以官方说明为准。

## 仓库内离线文件

- `third_party/dvsense_sdk/dvsense_driver-1.3.0-cp39-cp39-win_amd64.whl`：Python SDK 1.3.0，Windows x64 / CPython 3.9 专用。SHA-256：`3f704e73386e7aaf26681e736d73befc791201cc6212b02fce46a751d99c7476`，与 PyPI 1.3.0 对应文件公布值一致。
- `third_party/dvsense_sdk/DvsenseDriver-Setup-1.2.6.exe`：此前保存在本机的 Windows 安装程序，版本是 **1.2.6，不是 1.3.0**。SHA-256：`5f61646b437a6dbc00b1e3f8cba70f50776c0318341acacbca6d037f06ee3011`。该值只确认拷贝一致，未核实文件签名或来源；安装前请核对厂商信息，并注意版本兼容性。
- [`DVSLume 使用手册 PDF`](DVSLume使用手册%281%29.pdf)：本地保存的手册副本。SHA-256：`fa8315f70e8ce2210cec7f3996d4cca0ac2f6e2d214fda05211c2777d19eff17`。

在 Windows x64、CPython 3.9 环境中且已安装本项目依赖（尤其是 NumPy）时，可直接使用仓库内 wheel：

```powershell
python -m pip install --no-index --no-deps third_party/dvsense_sdk/dvsense_driver-1.3.0-cp39-cp39-win_amd64.whl
python tools/check_camera_sdk.py
# 或者：./scripts/INSTALL_CAMERA_SDK.ps1 -Offline
```

以上是 Python wheel，不替代其他 Python 版本、其他平台构建，也不保证包含完整 C++ SDK 或上位机。其他环境请从厂商渠道安装。`--no-deps` 不会安装缺失的依赖。

## 官方资料

- [SDK 安装与 Python 驱动](https://sdk.dvsense.com/zh/html/install_zh.html)
- [SDK 教程](https://sdk.dvsense.com/zh/html/tutorial_zh.html)
- [Python SDK 1.3.0 版本及各平台安装包](https://pypi.org/project/dvsense-driver/1.3.0/)
- [DVSense 官方 Python 示例仓库](https://github.com/dvsense/dvsense_driver_python_sample)
- [本项目双机接线记录](DVSLume_SYNC_CABLE_SPEC.md)：使用前对照对应相机型号和官方手册，切勿仅凭线色接线。

## 随仓库提供的示例及许可

`third_party/dvsense_driver_python_sample/` 是上述官方示例仓库在提交 `923943401c8f8d8a5581a6d559a6d7b1e48452ca` 的已跟踪示例快照，保留了上游 MIT 许可证与版权声明。示例许可不自动适用于 SDK 驱动、第三方动态库或厂商手册。

SDK 安装包包含厂商驱动及第三方组件；当前安装包元数据未注明再分发许可。官方示例的 MIT 许可证不自动适用于这些二进制文件或厂商手册。
