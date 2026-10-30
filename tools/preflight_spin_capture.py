"""Prepare and gate a dual-RAW segment before any spin solver is allowed to run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def longest_valid_segment(timestamps: np.ndarray, selected: np.ndarray,
                          maximum_gap_us: int, minimum_duration_us: int):
    valid = np.flatnonzero(np.asarray(selected) >= 0)
    if not len(valid):
        return None
    groups = np.split(valid, np.flatnonzero(np.diff(timestamps[valid]) > maximum_gap_us) + 1)
    candidates = []
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        duration = int(timestamps[last] - timestamps[first])
        if duration >= minimum_duration_us:
            candidates.append((duration, len(group), first, last))
    return max(candidates, default=None)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="双路 RAW 转速前置验收：轨迹、球内结构和人工复核包；不计算 RPM")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--start-offset-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/offline_fast_ball.yaml")
    parser.add_argument("--calibration", default="calibration/stereo_calibration.npz")
    parser.add_argument("--window-us", type=int, default=500)
    parser.add_argument("--annotation-frames", type=int, default=30)
    parser.add_argument("--maximum-gap-ms", type=float, default=20.0)
    parser.add_argument("--minimum-segment-ms", type=float, default=30.0)
    parser.add_argument("--reuse-path", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path_dir = output / "stereo_path"
    if not args.reuse_path:
        run([
            sys.executable, str(root / "tools" / "offline_stereo_path_audit.py"),
            "--master-raw", str(Path(args.master_raw).resolve()),
            "--slave-raw", str(Path(args.slave_raw).resolve()),
            "--config", str((root / args.config).resolve()),
            "--calibration", str((root / args.calibration).resolve()),
            "--start-offset-s", str(args.start_offset_s),
            "--duration-s", str(args.duration_s),
            "--bin-us", "1000", "--detection-step-us", "1000",
            "--output-dir", str(path_dir), "--no-preview",
        ])
    path_file = path_dir / "stereo_path.npz"
    if not path_file.exists():
        raise FileNotFoundError(f"Missing stereo path: {path_file}")
    path = np.load(path_file)
    segment = longest_valid_segment(
        path["timestamps_us"].astype(np.int64), path["selected"],
        round(args.maximum_gap_ms * 1000), round(args.minimum_segment_ms * 1000))
    if segment is None:
        report = {
            "status": "rejected_no_valid_stereo_motion_segment",
            "spin_solver_allowed": False,
            "path": str(path_file),
            "runtime_seconds": time.perf_counter() - started,
        }
        (output / "spin_preflight.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    duration_us, observations, first, last = segment
    packs = {}
    for camera, raw in (("master", args.master_raw), ("slave", args.slave_raw)):
        camera_output = output / f"marker_{camera}"
        run([
            sys.executable, str(root / "tools" / "visualize_monocular_observability.py"),
            "--raw", str(Path(raw).resolve()),
            "--path", str(path_dir / f"{camera}_path.npz"),
            "--first-index", str(first), "--last-index", str(last),
            "--windows-us", str(args.window_us), "--patch-size", "256",
            "--annotation-frames", str(args.annotation_frames),
            "--annotation-window-us", str(args.window_us),
            "--output-dir", str(camera_output),
        ])
        packs[camera] = camera_output / "annotation_pack" / "annotations.json"
    metrics_path = output / "marker_observability_metrics.json"
    run([
        sys.executable, str(root / "tools" / "audit_marker_observability.py"),
        "--pack", f"master={packs['master']}",
        "--pack", f"slave={packs['slave']}",
        "--output", str(metrics_path),
    ])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    automatic_pass = all(
        metrics["packs"][camera]["automatic_structure_gate"]["passed"]
        for camera in ("master", "slave"))
    report = {
        "status": ("requires_human_physical_marker_review" if automatic_pass else
                   "rejected_automatic_marker_structure"),
        "spin_solver_allowed": False,
        "reason": ("Automatic structure passed; human physical-marker centre/identity review remains required."
                   if automatic_pass else
                   "At least one camera failed ball-interior structure or polarity balance."),
        "segment": {
            "first_index": first, "last_index": last,
            "duration_ms": duration_us / 1000, "stereo_observations": observations,
        },
        "path": str(path_file),
        "marker_metrics": str(metrics_path),
        "human_review_contact_sheets": {
            camera: str((pack.parent / "annotation_contact_sheet.jpg").resolve())
            for camera, pack in packs.items()
        },
        "annotation_manifests": {camera: str(pack.resolve())
                                 for camera, pack in packs.items()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "spin_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
