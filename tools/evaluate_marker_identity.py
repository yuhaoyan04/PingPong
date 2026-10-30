"""Evaluate diagnostic candidate track IDs against human physical-dot identities."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_identity(payload: dict, candidate_by_frame: dict[int, list[dict]],
                      match_radius_px: float) -> tuple[dict, list[dict]]:
    """Compare candidate IDs with reviewed physical IDs in consecutive frames."""
    matches: dict[tuple[int, int], int] = {}
    matched_distances = []
    label_count = matched_label_count = 0
    reviewed_frames = 0
    for frame in payload["frames"]:
        if frame["status"] != "reviewed":
            continue
        reviewed_frames += 1
        frame_index = int(frame["local_path_index"])
        labels = frame["identity_labels"]
        candidates = candidate_by_frame.get(frame_index, [])
        label_count += len(labels)
        if not labels or not candidates:
            continue
        label_xy = np.asarray([[item["x"], item["y"]] for item in labels], np.float64)
        candidate_xy = np.asarray([[item["x"], item["y"]] for item in candidates], np.float64)
        distances = np.linalg.norm(label_xy[:, None, :] - candidate_xy[None, :, :], axis=2)
        label_indices, candidate_indices = linear_sum_assignment(distances)
        for label_index, candidate_index in zip(label_indices, candidate_indices):
            distance = float(distances[label_index, candidate_index])
            if distance > match_radius_px:
                continue
            physical_id = int(labels[label_index]["id"])
            candidate_id = int(candidates[candidate_index]["track_id"])
            matches[(frame_index, physical_id)] = candidate_id
            matched_distances.append(distance)
            matched_label_count += 1

    identity_pairs = retained_pairs = switched_pairs = unmatched_pairs = 0
    transition_rows = []
    reviewed = {int(frame["local_path_index"]): frame for frame in payload["frames"]
                if frame["status"] == "reviewed"}
    for frame_index in sorted(reviewed):
        next_index = frame_index + 1
        if next_index not in reviewed:
            continue
        current_ids = {int(item["id"]) for item in reviewed[frame_index]["identity_labels"]}
        next_ids = {int(item["id"]) for item in reviewed[next_index]["identity_labels"]}
        for physical_id in sorted(current_ids & next_ids):
            identity_pairs += 1
            first = matches.get((frame_index, physical_id))
            second = matches.get((next_index, physical_id))
            if first is None or second is None:
                unmatched_pairs += 1
                result = "unmatched"
            elif first == second:
                retained_pairs += 1
                result = "retained"
            else:
                switched_pairs += 1
                result = "switched"
            transition_rows.append({
                "frame_index": frame_index,
                "next_frame_index": next_index,
                "physical_id": physical_id,
                "candidate_track_id": first,
                "next_candidate_track_id": second,
                "result": result,
            })
    report = {
        "purpose": "candidate_track_identity_evaluation_against_human_physical_ids",
        "reviewed_frames": reviewed_frames,
        "unreviewed_or_unusable_frames": len(payload["frames"]) - reviewed_frames,
        "physical_label_observations": label_count,
        "matched_label_observations": matched_label_count,
        "framewise_label_recall": matched_label_count / max(1, label_count),
        "match_distance_median_px": (float(np.median(matched_distances))
                                     if matched_distances else None),
        "consecutive_physical_identity_pairs": identity_pairs,
        "retained_candidate_id_pairs": retained_pairs,
        "switched_candidate_id_pairs": switched_pairs,
        "unmatched_identity_pairs": unmatched_pairs,
        "identity_retention_fraction_all_pairs": retained_pairs / max(1, identity_pairs),
        "identity_switch_fraction_of_matched_pairs": (
            switched_pairs / max(1, retained_pairs + switched_pairs)),
        "accepted": False,
        "acceptance_gate": {
            "minimum_consecutive_physical_identity_pairs": 50,
            "minimum_framewise_label_recall": .80,
            "minimum_identity_retention_fraction_all_pairs": .90,
            "maximum_identity_switch_fraction_of_matched_pairs": .05,
        },
    }
    gate = report["acceptance_gate"]
    report["accepted"] = bool(
        identity_pairs >= gate["minimum_consecutive_physical_identity_pairs"]
        and report["framewise_label_recall"] >= gate["minimum_framewise_label_recall"]
        and report["identity_retention_fraction_all_pairs"] >=
            gate["minimum_identity_retention_fraction_all_pairs"]
        and report["identity_switch_fraction_of_matched_pairs"] <=
            gate["maximum_identity_switch_fraction_of_matched_pairs"])
    return report, transition_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估候选轨迹对物理黑点身份的保持率和换 ID 率")
    parser.add_argument("track_annotations")
    parser.add_argument("continuity_csv")
    parser.add_argument("--match-radius-px", type=float, default=10.0)
    parser.add_argument("--output", help="defaults beside track annotations")
    args = parser.parse_args()
    annotation_path = Path(args.track_annotations)
    output_path = (Path(args.output) if args.output else
                   annotation_path.parent / "marker_identity_evaluation.json")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    candidate_by_frame: dict[int, list[dict]] = defaultdict(list)
    with Path(args.continuity_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            candidate_by_frame[int(row["frame_index"])].append({
                "track_id": int(row["track_id"]),
                "x": float(row["x"]), "y": float(row["y"]),
            })

    report, transition_rows = evaluate_identity(
        payload, candidate_by_frame, args.match_radius_px)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_path.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = ["frame_index", "next_frame_index", "physical_id",
                  "candidate_track_id", "next_candidate_track_id", "result"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(transition_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["reviewed_frames"] == 0:
        raise SystemExit("没有已复核的连续身份标注")


if __name__ == "__main__":
    main()
