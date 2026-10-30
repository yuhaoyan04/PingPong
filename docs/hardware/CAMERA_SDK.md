# 动微相机 SDK 与资料

本项目此前实际使用 `dvsense-driver 1.3.0`、Windows x64、CPython 3.9 的 Python 驱动。官方包还有其他 Python 和平台的构建；使用前核对平台、Python 版本、相机型号和 SDK 兼容性。本仓库不包含本地虚拟环境，也没有把相机数据、序列号对应的采集文件或标定结果打包进来。

## 安装与检查

先按照根目录 README 创建并激活 Python 环境，再从官方 Python 包仓库安装本项目实验使用的版本：

```powershell
python -m pip install -r requirements-camera.txt
python tools/check_camera_sdk.py
```

Windows 用户也可以在激活环境后运行 `./scripts/INSTALL_CAMERA_SDK.ps1`；如需指定解释器，可传 `-Python <python路径>`。导入检查不会连接相机，也不能证明双机同步正常。安装驱动、上位机或解决 USB 设备权限时应参考厂商的系统安装说明；Linux 系统级库及支持版本也以官方说明为准。

## 官方资料

- [SDK 安装与 Python 驱动](https://sdk.dvsense.com/zh/html/install_zh.html)
- [SDK 教程](https://sdk.dvsense.com/zh/html/tutorial_zh.html)
- [Python SDK 1.3.0 版本及各平台安装包](https://pypi.org/project/dvsense-driver/1.3.0/)
- [DVSense 官方 Python 示例仓库](https://github.com/dvsense/dvsense_driver_python_sample)
- [本项目双机接线记录](DVSLume_SYNC_CABLE_SPEC.md)：使用前对照对应相机型号和官方手册，切勿仅凭线色接线。

## 随仓库提供的示例及许可

`third_party/dvsense_driver_python_sample/` 是上述官方示例仓库在提交 `923943401c8f8d8a5581a6d559a6d7b1e48452ca` 的已跟踪示例快照，保留了上游 MIT 许可证与版权声明。示例许可不自动适用于 SDK 驱动、第三方动态库或厂商手册。

SDK 的本机安装目录包含厂商驱动和第三方动态库；当前未找到明确允许在本公共仓库再分发整包二进制的许可，因而未把二进制和厂商 PDF 直接复制到仓库。请使用上面的官方安装渠道；如厂商提供了适用的再分发许可，可根据许可另行决定如何提供离线包。
