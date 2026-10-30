"""Dry-run optical-flow marker propagation without creating physical IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from annotate_dual_marker_identities import propagate_labels_optical_flow


def load_proposals(annotation_path: Path) -> dict[str, list[dict]]:
    payload = json.loads((annotation_path.parent / "marker_region_proposals.json").read_text(
        encoding="utf-8"))
    return {record["frame"]: record.get("proposals", [])
            for record in payload["records"]}


def audit_camera(annotation_path: Path) -> dict:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    frames = annotations["frames"]
    proposals = load_proposals(annotation_path)
    first = proposals.get(frames[0]["frame"], [])
    labels = [{"id": index + 1, "x": item["x"], "y": item["y"]}
              for index, item in enumerate(first)]
    previous = cv2.imread(str(annotation_path.parent / frames[0]["frame"]))
    records = [{
        "frame": frames[0]["frame"], "timestamp_us": frames[0]["timestamp_us"],
        "labels": len(labels), "median_confidence": 1.0,
    }]
    for frame in frames[1:]:
        current = cv2.imread(str(annotation_path.parent / frame["frame"]))
        labels = propagate_labels_optical_flow(
            previous, current, labels, proposals.get(frame["frame"], []))
        records.append({
            "frame": frame["frame"], "timestamp_us": frame["timestamp_us"],
            "labels": len(labels),
            "median_confidence": (float(np.median([
                item["confidence"] for item in labels])) if labels else None),
        })
        previous = current
        if not labels:
            break
    counts = np.asarray([item["labels"] for item in records], np.int32)
    return {
        "seed_markers": len(first),
        "processed_frames": len(records),
        "frames_with_at_least_3_proposals": int(np.count_nonzero(counts >= 3)),
        "last_frame_marker_count": int(counts[-1]),
        "records": records,
        "interpretation": (
            "diagnostic proposal survival only; IDs are arbitrary and are not physical identity labels"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "diagnostic_only",
        "master": audit_camera(args.master_annotations),
        "slave": audit_camera(args.slave_annotations),
        "physical_identity_proven": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
