"""Evaluate anonymous marker-region proposals against manually reviewed centres."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="用人工点标注评估候选检出率和定位误差")
    parser.add_argument("annotations")
    parser.add_argument("--proposals",
                        help="defaults to marker_region_proposals.json next to annotations")
    parser.add_argument("--match-radius-px", type=float, default=10.0)
    parser.add_argument("--output", help="defaults to marker_proposal_evaluation.json")
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    proposal_path = (Path(args.proposals) if args.proposals else
                     annotation_path.parent / "marker_region_proposals.json")
    output_path = (Path(args.output) if args.output else
                   annotation_path.parent / "marker_proposal_evaluation.json")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal_lookup = {record["frame"]: record for record in proposals["records"]}

    true_positive = false_positive = false_negative = 0
    distances = []
    rows = []
    reviewed = 0
    unusable = 0
    for frame in annotations["frames"]:
        if frame["status"] == "unusable":
            unusable += 1
            continue
        if frame["status"] != "reviewed":
            continue
        reviewed += 1
        labels = np.asarray(frame["labels"], np.float64).reshape(-1, 2)
        proposed_values = proposal_lookup.get(frame["frame"], {}).get("proposals", [])
        points = np.asarray([[item["x"], item["y"]] for item in proposed_values],
                            np.float64).reshape(-1, 2)
        matches = []
        if len(labels) and len(points):
            matrix = np.linalg.norm(labels[:, None, :] - points[None, :, :], axis=2)
            label_indices, point_indices = linear_sum_assignment(matrix)
            matches = [(int(i), int(j), float(matrix[i, j]))
                       for i, j in zip(label_indices, point_indices)
                       if matrix[i, j] <= args.match_radius_px]
        tp = len(matches)
        fp = len(points) - tp
        fn = len(labels) - tp
        true_positive += tp
        false_positive += fp
        false_negative += fn
        distances.extend(value[2] for value in matches)
        rows.append({
            "frame": frame["frame"],
            "labels": len(labels),
            "proposals": len(points),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "mean_localization_error_px": (float(np.mean([value[2] for value in matches]))
                                            if matches else ""),
        })
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    report = {
        "purpose": "framewise_anonymous_marker_region_proposal_evaluation",
        "reviewed_frames": reviewed,
        "unusable_frames": unusable,
        "unreviewed_frames": len(annotations["frames"]) - reviewed - unusable,
        "match_radius_px": args.match_radius_px,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "localization_error_median_px": (float(np.median(distances)) if distances else None),
        "localization_error_p90_px": (float(np.percentile(distances, 90)) if distances else None),
        "accepted_for_identity_or_spin": False,
        "warning": "Framewise detection accuracy does not validate cross-frame physical identity.",
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "frame", "labels", "proposals", "true_positive", "false_positive",
            "false_negative", "mean_localization_error_px"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if reviewed == 0:
        raise SystemExit("没有 status=reviewed 的人工标注帧；报告只记录空评估")


if __name__ == "__main__":
    main()
