"""Print DVSLume tool parameters without changing camera settings."""

from dvsense_driver.camera_manager import DvsCameraManager
from dvsense_driver.hal.camera_tool import ToolType

manager = DvsCameraManager()
manager.update_cameras()
for desc in manager.get_camera_descs():
    camera = manager.open_camera(desc.serial)
    print(f"\n{desc.serial}")
    for name in ["BIAS", "ANTI_FLICKER", "EVENT_TRAIL_FILTER", "EVENT_RATE_CONTROL", "ROI"]:
        tool_type = getattr(ToolType, name)
        try:
            tool = camera.get_tool(tool_type)
            print(name)
            for param in tool.get_all_param_info():
                ok_info, info = tool.get_param_info(param)
                ok_value, value = tool.get_param(param)
                details = {}
                if ok_info:
                    for attr in ("min", "max", "default_value", "unit", "options"):
                        if hasattr(info, attr):
                            details[attr] = getattr(info, attr)
                print(" ", param, "=", value if ok_value else "?", ";", details)
        except Exception as exc:
            print(name, "不可用:", exc)
