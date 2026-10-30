"""Fail-closed identity evaluation, physical-template build, and spin validation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]


def annotation_readiness(payload: dict) -> dict:
    reviewed = [frame for frame in payload.get("frames", [])
                if frame.get("status") == "reviewed"]
    shared_counts = []
    for frame in reviewed:
        master = {int(item["id"]) for item in frame.get("master_labels", [])}
        slave = {int(item["id"]) for item in frame.get("slave_labels", [])}
        shared_counts.append(len(master & slave))
    shared = int(sum(shared_counts))
    connected = int(sum(count >= 3 for count in shared_counts))
    checks = {
        "reviewed_frames": len(reviewed) >= 20,
        "shared_identity_observations": shared >= 60,
        "frames_with_at_least_3_shared_ids": connected >= 10,
    }
    return {
        "ready": all(checks.values()),
        "reviewed_frames": len(reviewed),
        "shared_identity_observations": shared,
        "frames_with_at_least_3_shared_ids": connected,
        "thresholds": {
            "reviewed_frames": 20,
            "shared_identity_observations": 60,
            "frames_with_at_least_3_shared_ids": 10,
        },
        "checks": checks,
    }


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键评估双路物理 ID、构建真实球模板并可选运行独立转速验证")
    parser.add_argument("--dual-annotations", type=Path, required=True)
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--master-continuity", type=Path, required=True)
    parser.add_argument("--slave-continuity", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration/stereo_calibration.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-marker-sets", type=Path)
    parser.add_argument("--allow-calibration-overlap", action="store_true")
    parser.add_argument("--watch", action="store_true",
                        help="wait for the annotation gate, then finalize automatically")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    final_report = output_dir / "physical_template_finalization.json"
    payload = json.loads(args.dual_annotations.read_text(encoding="utf-8"))
    readiness = annotation_readiness(payload)
    report = {
        "status": "awaiting_review" if not readiness["ready"] else "running",
        "spin_solver_allowed": False,
        "annotation_readiness": readiness,
        "identity_evaluation": None,
        "physical_template": None,
        "spin_validation": None,
    }
    if not readiness["ready"] and args.watch:
        report["status"] = "watching_annotations"
        report["reason"] = "waiting for the human-reviewed shared physical ID gate"
        write_report(final_report, report)
        previous = readiness
        while not readiness["ready"]:
            time.sleep(max(args.poll_seconds, .25))
            try:
                payload = json.loads(args.dual_annotations.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            readiness = annotation_readiness(payload)
            if readiness != previous:
                report["annotation_readiness"] = readiness
                write_report(final_report, report)
                previous = readiness
        report["status"] = "running"
        report["annotation_readiness"] = readiness
        report["reason"] = "annotation gate reached; running finalization"
        write_report(final_report, report)
    elif not readiness["ready"]:
        report["reason"] = "human-reviewed shared physical IDs have not reached the gate"
        write_report(final_report, report)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "dual_identity_evaluation.json"
    run([
        sys.executable, str(ROOT / "tools" / "evaluate_dual_marker_identities.py"),
        str(args.dual_annotations), str(args.master_continuity),
        str(args.slave_continuity), "--output", str(identity_path),
    ])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    report["identity_evaluation"] = identity
    if not identity.get("accepted", False):
        report["status"] = "rejected_identity_gate"
        report["reason"] = identity.get("reason")
        write_report(final_report, report)
        return

    template_path = output_dir / "physical_ball_template.npz"
    run([
        sys.executable, str(ROOT / "tools" / "build_labeled_ball_template.py"),
        "--dual-annotations", str(args.dual_annotations),
        "--master-annotations", str(args.master_annotations),
        "--slave-annotations", str(args.slave_annotations),
        "--paths", str(args.paths), "--calibration", str(args.calibration),
        "--identity-evaluation", str(identity_path), "--output", str(template_path),
    ])
    template_report_path = template_path.with_suffix(".json")
    template = json.loads(template_report_path.read_text(encoding="utf-8"))
    report["physical_template"] = template
    if not template.get("spin_solver_allowed", False):
        report["status"] = "rejected_template_gate"
        report["reason"] = template.get("reason")
        write_report(final_report, report)
        return

    if args.validation_marker_sets:
        validation_path = output_dir / "calibrated_template_spin.json"
        command = [
            sys.executable, str(ROOT / "tools" / "solve_calibrated_template_spin.py"),
            "--template", str(template_path),
            "--template-report", str(template_report_path),
            "--marker-sets", str(args.validation_marker_sets),
            "--output", str(validation_path),
        ]
        if args.allow_calibration_overlap:
            command.append("--allow-calibration-overlap")
        run(command)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        report["spin_validation"] = validation
        if validation.get("spin_rpm") is None:
            report["status"] = "rejected_spin_validation"
            report["reason"] = validation.get("reason")
            write_report(final_report, report)
            return

    report["status"] = "accepted_template"
    report["spin_solver_allowed"] = True
    report["reason"] = "identity and physical-template gates passed"
    write_report(final_report, report)


if __name__ == "__main__":
    main()
