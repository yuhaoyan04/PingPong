"""Run the auditable trajectory and multi-baseline spin pipeline for one trial."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def split_motion_segments(timestamps_us: np.ndarray, selected: np.ndarray,
                          maximum_gap_us: int, minimum_duration_us: int):
    valid = np.flatnonzero(np.asarray(selected) >= 0)
    if not len(valid):
        return []
    breaks = np.flatnonzero(np.diff(timestamps_us[valid]) > maximum_gap_us)
    groups = np.split(valid, breaks + 1)
    segments = []
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        duration = int(timestamps_us[last] - timestamps_us[first])
        if duration >= minimum_duration_us:
            segments.append((first, last, duration, len(group)))
    return segments


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="一条命令完成真实球轨迹、球速和转速质量验收")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--start-offset-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/offline_fast_ball.yaml")
    parser.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    parser.add_argument("--table-frame", help="calibration/table_frame.npz；提供后输出旋转类型")
    parser.add_argument("--bin-us", type=int, default=1000)
    parser.add_argument("--path-detection-step-us", type=int, default=1000,
                        help="Formal validation requires 1000; larger values are diagnostic only")
    parser.add_argument("--maximum-gap-ms", type=float, default=20.0)
    parser.add_argument("--minimum-segment-ms", type=float, default=60.0)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--max-features", type=int, default=160)
    parser.add_argument("--fast", action="store_true",
                        help="快速诊断；正式验收请不要使用")
    parser.add_argument("--reuse-path", action="store_true")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip MP4 rendering to reduce post-processing latency")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    path_dir = output / "stereo_path"
    if not args.reuse_path:
        path_command = [sys.executable, str(root / "tools" / "offline_stereo_path_audit.py"),
             "--master-raw", str(Path(args.master_raw).resolve()),
             "--slave-raw", str(Path(args.slave_raw).resolve()),
             "--config", str((root / args.config).resolve()),
             "--calibration", str((root / args.calibration).resolve()),
             "--start-offset-s", str(args.start_offset_s),
             "--duration-s", str(args.duration_s), "--bin-us", str(args.bin_us),
             "--detection-step-us", str(args.path_detection_step_us),
             "--output-dir", str(path_dir)]
        if args.no_preview:
            path_command.append("--no-preview")
        run(path_command)
    path_file = path_dir / "stereo_path.npz"
    if not path_file.exists():
        raise FileNotFoundError(f"Missing fitted path: {path_file}")
    path = np.load(path_file)
    segments = split_motion_segments(
        path["timestamps_us"], path["selected"], round(args.maximum_gap_ms * 1000),
        round(args.minimum_segment_ms * 1000))[:max(1, args.max_segments)]
    if not segments:
        raise RuntimeError("No sufficiently long stereo motion segment was found")

    reports = []
    for number, (first, last, duration_us, observations) in enumerate(segments, 1):
        segment_dir = output / f"segment_{number:02d}_{first}_{last}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        lags = sorted(set(max(1, value) for value in args.lags))
        spin_jsons = [segment_dir / f"spin_lag{lag}.json" for lag in lags]
        command = [
            sys.executable, str(root / "tools" / "offline_pooled_spin.py"),
            "--master-raw", str(Path(args.master_raw).resolve()),
            "--slave-raw", str(Path(args.slave_raw).resolve()),
            "--paths", str(path_dir), "--calibration", str((root / args.calibration).resolve()),
            "--first-index", str(first), "--last-index", str(last),
            "--lag-frames", *map(str, lags), "--max-features", str(args.max_features),
            "--scale-angular-threshold",
            "--output", str(segment_dir / "spin_multilag.json")]
        if args.fast:
            command.append("--fast")
        run(command)
        report_path = segment_dir / "measurement.json"
        states_path = segment_dir / "states.csv"
        preview_path = segment_dir / "measurement.mp4"
        summarize_command = [sys.executable, str(root / "tools" / "summarize_pooled_spin.py"),
             "--spin-json", *map(str, spin_jsons), "--stereo-path", str(path_file),
             "--first-index", str(first), "--last-index", str(last),
             "--output", str(report_path), "--states-csv", str(states_path)]
        if not args.no_preview:
            summarize_command += ["--path-video", str(path_dir / "stereo_path.mp4"),
                                  "--preview", str(preview_path)]
        if args.table_frame:
            summarize_command += ["--table-frame", str(Path(args.table_frame).resolve())]
        run(summarize_command)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append({"segment": number, "first_index": first, "last_index": last,
                        "duration_ms": duration_us / 1000, "stereo_observations": observations,
                        "status": report["status"], "speed_mps": report["speed_mps_median"],
                        "spin_rpm": report["spin_rpm"], "report": str(report_path)})
    manifest = {"mode": "fast_diagnostic" if args.fast else "full_quality",
                "path": str(path_file), "segments": reports,
                "total_runtime_seconds": time.perf_counter() - started}
    manifest_path = output / "trial_measurement.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
