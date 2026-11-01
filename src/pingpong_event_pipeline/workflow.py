"""One Linux/Windows CLI for acquisition through YOLO tracking and offline spin."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("PINGPONG_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATASET = Path(os.environ.get("PINGPONG_DATASET", ROOT / "data/server_marker_two_camera_train_v1")).resolve()
EXPERIMENT = Path(os.environ.get("PINGPONG_EXPERIMENT", ROOT / "experiments/two_camera_v1")).resolve()
TRACKS = ROOT / "experiments/tracking_min3_v1/filtered/rotation_tracks_all.csv"
STAGES = {
    "sdk-check": "tools/check_camera_sdk.py",
    "sdk-audit": "tools/audit_camera_runtime.py",
    "event-timing": "tools/audit_event_timing.py",
    "spin-check": "tools/review_spin_acceptance.py",
    "serve-spin": "tools/review_serve_spin.py",
    "radius-spin": "tools/review_spin_radii.py",
    "clip-sync": "tools/review_clip_sync.py",
    "ball-raw-audit": "tools/review_ball_raw_sync.py",
    "ball-raw-video": "tools/render_ball_raw_review.py",
    "ball-raw-package": "tools/finalize_ball_raw_review.py",
    "event-clock-runs": "tools/export_event_clock_runs.py",
    "appearance-period": "tools/review_appearance_period.py",
    "raw-radius-review": "tools/review_raw_ball_radii.py",
    "smooth-tracks": "tools/review_track_smoothing.py",
    "train-holdout": "tools/yolo/audit_group_holdout.py",
    "system-audit": "tools/audit_system_iteration.py",
    "system-review": "tools/render_system_iteration.py",
    "refine-observations": "tools/refine_observations.py",
    "refine-continuity": "tools/refine_track_continuity.py",
    "review-observations": "tools/review_spin_direction.py",
    "capture": "tools/dual_capture_isolated.py",
    "camera-tune": "tools/dual_camera_tuner.py",
    "calibration-capture": "tools/manual_calibration_capture.py",
    "calibrate": "tools/calibrate_stereo.py",
    "validate-calibration": "tools/validate_stereo_calibration.py",
    "legacy-nine-serve-candidates": "tools/batch_marker_crops.py",
    "ball-track": "tools/track_monocular_ball_circle.py",
    "prepare-group": "tools/prepare_raw_group.py",
    "ball-refine": "tools/run_ball_pipeline.py",
    "ball-crops": "tools/build_marker_label_dataset.py",
    "enhance": "tools/export_supported_ball_crops.py",
    "annotate": "tools/marker_annotation_player.py",
    "export-yolo": "tools/export_marker_yolo.py",
    "package": "tools/package_two_camera_training.py",
    "verify-package": "tools/verify_two_camera_package.py",
    "audit": "tools/yolo/audit_latest.py",
    "train": "tools/yolo/train_camera.py",
    "infer": "tools/yolo/apply_model.py",
    "filter": "tools/yolo/filter_track_sequences.py",
}


def environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("MPLBACKEND", "Agg")
    return env


def run_stage(stage, arguments):
    return subprocess.run([sys.executable, str(ROOT / STAGES[stage]), *arguments],
                          cwd=ROOT, env=environment(), check=True)


def doctor():
    versions = {}
    for name in ("numpy", "scipy", "cv2", "numba", "torch", "ultralytics", "PyQt5"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "importable")
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    from .marker_spin import sha256
    models = {}
    for camera in ("F3E", "F59"):
        path = EXPERIMENT / camera / "train/weights/last.pt"
        models[camera] = {"path": str(path), "exists": path.is_file(),
                          "sha256": sha256(path) if path.is_file() else None}
    sdk_error = None
    try:
        importlib.import_module("dvsense_driver.raw_file_reader")
        importlib.import_module("dvsense_driver.camera_manager")
        sdk = True
    except Exception as error:
        sdk, sdk_error = False, str(error)
    calibration = list((ROOT / "calibration").glob("*.npz")) + list((ROOT / "data/calibration").glob("*.npz"))
    return {"project_root": str(ROOT), "python": sys.executable, "versions": versions,
            "sdk_importable": sdk, "camera_connection_verified": False,
            "sdk_import_error": sdk_error,
            "ffmpeg": shutil.which("ffmpeg"), "dataset_present": DATASET.is_dir(),
            "consecutive_tracks_present": TRACKS.is_file(), "models": models,
            "calibration_files": [str(p) for p in calibration],
            "raw_files": [str(p) for p in (ROOT / "data").rglob("*.raw")],
            "available_offline_path": "latest crops -> camera YOLO -> consecutive tracks -> conditional spin",
            "hardware_path": "SDK and actual cameras are required for capture/RAW replay; not exercised by doctor"}


def spin(tracks, dataset, output, config_path, review=True):
    from .marker_spin import estimate_tracks, write_outputs
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    report = estimate_tracks(tracks, dataset, config)
    report["input_tracks"] = str(Path(tracks).resolve())
    report["dataset"] = str(Path(dataset).resolve())
    write_outputs(report, output)
    if review:
        from .spin_review import create_review
        create_review(report, tracks, dataset, output)
    print(json.dumps({"output": str(Path(output).resolve()), "status": report["status"],
                      "windows": report["window_count"], "usable_windows": report["usable_window_count"],
                      "spin_rpm": None, "physical_measurement_validated": False}, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="PingPong: SDK / 数据 / YOLO / 追踪 / 离线转速")
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("doctor", help="检查当前环境、模型、SDK 和输入数据")
    health.add_argument("--output", type=Path)
    stage = sub.add_parser("run", help="执行流水线环节；例如 run infer --help")
    stage.add_argument("stage", choices=STAGES)
    stage.add_argument("arguments", nargs=argparse.REMAINDER)
    estimate = sub.add_parser("spin", help="对已有连续轨迹估计转速候选")
    estimate.add_argument("--tracks", type=Path, default=TRACKS)
    estimate.add_argument("--dataset", type=Path, default=DATASET)
    estimate.add_argument("--config", type=Path, default=ROOT / "config/offline_marker_spin.json")
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument("--no-review", action="store_true")
    motion = sub.add_parser("motion", help="生成球心/标记轨迹、球速、转速的统一可视化")
    motion.add_argument("--tracking", type=Path, default=ROOT / "output/latest/tracking")
    motion.add_argument("--spin-report", type=Path, default=ROOT / "output/latest/spin/spin_report.json")
    motion.add_argument("--dataset", type=Path, default=DATASET)
    motion.add_argument("--output", type=Path, required=True)
    offline = sub.add_parser("offline", aliases=["pipeline"], help="现有模型推理 -> 三帧筛选 -> 转速/球速/轨迹可视化")
    offline.add_argument("--camera", choices=["F3E", "F59", "all"], default="all")
    offline.add_argument("--device", default="0")
    offline.add_argument("--replay-cache", action="store_true", help="重放已有检测缓存；不作为新模型推理")
    offline.add_argument("--source", type=Path, help="自定义数据包内的组目录，需选单相机")
    offline.add_argument("--output", type=Path, default=ROOT / "output" / ("offline_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    for command in (motion, offline):
        command.add_argument("--ball-diameter-m", type=float, default=.040)
        command.add_argument("--calibration", type=Path, help="真实标定 NPZ；T 单位为米")
        command.add_argument("--master-camera", choices=["F3E", "F59"], help="标定 K1 对应的相机，必须显式指定")
        command.add_argument("--slave-offset-us", type=int, default=0, help="加入从机时间戳的校正量")
        command.add_argument("--no-videos", action="store_true")
        command.add_argument("--runtime-report", type=Path, default=ROOT / "output/sdk_validation_v2/report.json")
    args = parser.parse_args(argv)
    if args.command in {"motion", "offline", "pipeline"} and args.calibration and not args.master_camera:
        parser.error("--calibration requires explicit --master-camera")
    if args.command == "doctor":
        report = doctor()
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
    elif args.command == "run":
        run_stage(args.stage, args.arguments)
    elif args.command == "spin":
        spin(args.tracks, args.dataset, args.output, args.config, not args.no_review)
    elif args.command == "motion":
        from .motion_review import build_report
        report = build_report(args.dataset, args.tracking, args.spin_report, args.output,
                              args.ball_diameter_m, args.calibration, args.master_camera,
                              args.slave_offset_us, not args.no_videos, args.runtime_report)
        print(json.dumps({"output": str(args.output.resolve()), "frames": report["ball_samples"], "review": "review.html"}))
    elif args.command in {"offline", "pipeline"}:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        arguments = ["--camera", args.camera, "--device", args.device, "--output", str(args.output / "tracking")]
        if args.source:
            arguments += ["--source", str(args.source.resolve())]
        if args.replay_cache:
            arguments += ["--replay-cache"]
        run_stage("infer", arguments)
        spin(args.output / "tracking/rotation_tracks_all.csv", DATASET, args.output / "spin",
             ROOT / "config/offline_marker_spin.json")
        from .motion_review import build_report
        build_report(DATASET, args.output / "tracking", args.output / "spin/spin_report.json", args.output / "motion",
                     args.ball_diameter_m, args.calibration, args.master_camera, args.slave_offset_us,
                     not args.no_videos, args.runtime_report)
        (args.output / "run.json").write_text(json.dumps({
            "camera": args.camera, "device": args.device, "replayed_cached_detections": args.replay_cache,
            "minimum_consecutive_frames": 3, "spin_report": "spin/spin_report.json",
            "review": "motion/review.html", "spin_review": "spin/review.html",
            "motion_report": "motion/motion_report.json", "physical_measurement_validated": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
