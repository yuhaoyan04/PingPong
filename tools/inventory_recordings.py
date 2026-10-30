"""Inventory recorded RAW pairs without interpreting their visual contents."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pingpong_event_pipeline.acquisition import spin_capture_config_valid


FIELDS = [
    "run_id", "master_raw", "slave_raw", "total_size_mb", "duration_s",
    "master_events", "slave_events", "master_saturation_fraction",
    "slave_saturation_fraction", "capture_complete", "metadata_status",
    "bias_diff_on", "bias_diff_off", "trail_filter_enabled",
    "event_rate_control_enabled", "spin_capture_config_valid",
]


def acquisition_audit(capture: dict) -> dict:
    config = capture.get("acquisition_config", capture)
    required = ("bias_diff_on", "bias_diff_off", "trail_filter_enabled",
                "event_rate_control_enabled")
    known = all(name in config for name in required)
    return {
        "bias_diff_on": config.get("bias_diff_on", ""),
        "bias_diff_off": config.get("bias_diff_off", ""),
        "trail_filter_enabled": config.get("trail_filter_enabled", ""),
        "event_rate_control_enabled": config.get("event_rate_control_enabled", ""),
        "spin_capture_config_valid": spin_capture_config_valid(config),
        "known": known,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", default="output/ball_trials")
    parser.add_argument("--output-dir", default="output/research")
    args = parser.parse_args()
    root = Path(args.recordings_dir)
    rows = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        raws = sorted(run.glob("*.raw"))
        if not raws:
            continue
        master = next((path for path in raws if "master" in path.name.lower()), None)
        slave = next((path for path in raws if "slave" in path.name.lower()), None)
        capture_path = run / "capture.json"
        capture = json.loads(capture_path.read_text(encoding="utf-8")) if capture_path.exists() else {}
        stats = capture.get("capture_stats", {})
        acquisition = acquisition_audit(capture)
        if not stats:
            metadata_status = "capture_stats_missing"
        elif not acquisition["known"]:
            metadata_status = "capture_stats_complete_acquisition_unknown"
        else:
            metadata_status = "complete"
        row = {
            "run_id": run.name,
            "master_raw": str(master.resolve()) if master else "",
            "slave_raw": str(slave.resolve()) if slave else "",
            "total_size_mb": round(sum(path.stat().st_size for path in raws) / 2**20, 3),
            "duration_s": capture.get("duration_s", ""),
            "master_events": stats.get("master", {}).get("events", ""),
            "slave_events": stats.get("slave", {}).get("events", ""),
            "master_saturation_fraction": stats.get("master", {}).get("saturation_fraction", ""),
            "slave_saturation_fraction": stats.get("slave", {}).get("saturation_fraction", ""),
            "capture_complete": capture.get("completed", ""),
            "metadata_status": metadata_status,
            **{key: value for key, value in acquisition.items() if key != "known"},
        }
        rows.append(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "recording_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    report = {"recordings_dir": str(root.resolve()), "recording_count": len(rows), "recordings": rows}
    (output / "recording_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
