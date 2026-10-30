"""Exercise official SDK imports, device enumeration and optional RAW replay.

No camera is opened or recording started. RAW evidence used here may be a
clock test, and is never presented as an actual ball/YOLO measurement.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--probe-devices", action="store_true")
    parser.add_argument("--sample-us", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "sdk_version": None, "sdk_import_ok": False, "device_enumeration_attempted": args.probe_devices,
              "devices": [], "raw_replay_ok": False, "physical_capture_verified": False,
              "training_recording_timebase_verified": False, "errors": []}
    try:
        result["sdk_version"] = importlib.metadata.version("dvsense-driver")
        for name in ["dvsense_driver.camera_manager", "dvsense_driver.raw_file_reader", "dvsense_driver.hal.camera_tool"]:
            importlib.import_module(name)
        result["sdk_import_ok"] = True
    except Exception as error:
        result["errors"].append(f"SDK import: {error}")
    if result["sdk_import_ok"] and args.probe_devices:
        try:
            from dvsense_driver.camera_manager import DvsCameraManager
            manager = DvsCameraManager()
            manager.update_cameras()
            result["devices"] = [str(d.serial) for d in manager.get_camera_descs()]
        except Exception as error:
            result["errors"].append(f"device enumeration: {error}")
    if result["sdk_import_ok"] and args.raw:
        try:
            import cv2
            import numpy as np
            from pingpong_event_pipeline.events import RawWindowReader, sae, render_bgr
            raw = RawWindowReader(args.raw)
            start = raw.info.start_us
            events = raw.window(start, min(start + args.sample_us, raw.info.end_us))
            times = events["timestamp"].astype(np.int64)
            monotonic = bool(np.all(np.diff(times) >= 0))
            in_bounds = bool(np.all((events["x"] < raw.info.width) & (events["y"] < raw.info.height)))
            result["raw"] = {"path": str(args.raw.resolve()), "sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                             "width": raw.info.width, "height": raw.info.height,
                             "file_start_us": raw.info.start_us, "file_end_us": raw.info.end_us,
                             "sample_window_us": args.sample_us, "sample_events": len(events),
                             "sample_monotonic": monotonic, "sample_coordinates_in_bounds": in_bounds,
                             "scope": "small RAW decoder smoke test; not ball measurement or full-file timing validation"}
            result["raw_replay_ok"] = len(events) > 0 and monotonic and in_bounds
            image = render_bgr(sae(events, raw.info.width, raw.info.height,
                                   start + args.sample_us, args.sample_us / 2))
            cv2.imwrite(str(args.output / "raw_preview.png"), image)
        except Exception as error:
            result["errors"].append(f"RAW replay: {error}")
    (args.output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["errors"] or not result["sdk_import_ok"] or (args.raw and not result["raw_replay_ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
