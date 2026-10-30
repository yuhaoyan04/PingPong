"""Evaluate temporal identity tracking in both cameras against shared physical IDs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_candidates(path: Path) -> Dict[int, List[dict]]:
    by_timestamp: Dict[int, List[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_timestamp[int(row["timestamp_us"])].append({
                "track_id": int(row["track_id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
            })
    return by_timestamp


def evaluate_side(frames: List[dict], label_key: str,
                  candidates: Dict[int, List[dict]], match_radius_px: float) -> dict:
    reviewed = [(position, frame) for position, frame in enumerate(frames)
                if frame["status"] == "reviewed"]
    matches = {}
    distances = []
    label_count = matched_count = 0
    for _, frame in reviewed:
        timestamp = int(frame["timestamp_us"])
        labels = frame[label_key]
        observed = candidates.get(timestamp, [])
        label_count += len(labels)
        if not labels or not observed:
            continue
        label_xy = np.asarray([[item["x"], item["y"]] for item in labels], np.float64)
        candidate_xy = np.asarray([[item["x"], item["y"]] for item in observed], np.float64)
        costs = np.linalg.norm(label_xy[:, None, :] - candidate_xy[None, :, :], axis=2)
        label_indices, candidate_indices = linear_sum_assignment(costs)
        for label_index, candidate_index in zip(label_indices, candidate_indices):
            distance = float(costs[label_index, candidate_index])
            if distance > match_radius_px:
                continue
            physical_id = int(labels[label_index]["id"])
            matches[(timestamp, physical_id)] = int(observed[candidate_index]["track_id"])
            distances.append(distance)
            matched_count += 1

    identity_pairs = retained = switched = unmatched = 0
    for (first_position, first), (second_position, second) in zip(reviewed[:-1], reviewed[1:]):
        if second_position != first_position + 1:
            continue
        first_ids = {int(item["id"]) for item in first[label_key]}
        second_ids = {int(item["id"]) for item in second[label_key]}
        for physical_id in first_ids & second_ids:
            identity_pairs += 1
            first_track = matches.get((int(first["timestamp_us"]), physical_id))
            second_track = matches.get((int(second["timestamp_us"]), physical_id))
            if first_track is None or second_track is None:
                unmatched += 1
            elif first_track == second_track:
                retained += 1
            else:
                switched += 1
    return {
        "physical_label_observations": label_count,
        "matched_label_observations": matched_count,
        "framewise_label_recall": matched_count / max(1, label_count),
        "match_distance_median_px": float(np.median(distances)) if distances else None,
        "consecutive_physical_identity_pairs": identity_pairs,
        "retained_candidate_id_pairs": retained,
        "switched_candidate_id_pairs": switched,
        "unmatched_identity_pairs": unmatched,
        "identity_retention_fraction_all_pairs": retained / max(1, identity_pairs),
        "identity_switch_fraction_of_matched_pairs": switched / max(1, retained + switched),
    }


def evaluate(payload: dict, master_candidates: Dict[int, List[dict]],
             slave_candidates: Dict[int, List[dict]], match_radius_px: float = 10.0) -> dict:
    frames = payload["frames"]
    reviewed = [frame for frame in frames if frame["status"] == "reviewed"]
    shared_counts = []
    for frame in reviewed:
        master_ids = {int(item["id"]) for item in frame["master_labels"]}
        slave_ids = {int(item["id"]) for item in frame["slave_labels"]}
        shared_counts.append(len(master_ids & slave_ids))
    master = evaluate_side(frames, "master_labels", master_candidates, match_radius_px)
    slave = evaluate_side(frames, "slave_labels", slave_candidates, match_radius_px)
    gate = {
        "minimum_reviewed_frames": 20,
        "minimum_shared_stereo_identity_observations": 60,
        "minimum_consecutive_physical_identity_pairs_per_camera": 50,
        "minimum_framewise_label_recall_per_camera": .80,
        "minimum_identity_retention_fraction_per_camera": .90,
        "maximum_identity_switch_fraction_of_matched_pairs_per_camera": .05,
    }
    shared_observations = int(sum(shared_counts))
    per_side_pass = []
    for side in (master, slave):
        per_side_pass.append(bool(
            side["consecutive_physical_identity_pairs"] >=
                gate["minimum_consecutive_physical_identity_pairs_per_camera"]
            and side["framewise_label_recall"] >=
                gate["minimum_framewise_label_recall_per_camera"]
            and side["identity_retention_fraction_all_pairs"] >=
                gate["minimum_identity_retention_fraction_per_camera"]
            and side["identity_switch_fraction_of_matched_pairs"] <=
                gate["maximum_identity_switch_fraction_of_matched_pairs_per_camera"]))
    accepted = bool(
        len(reviewed) >= gate["minimum_reviewed_frames"]
        and shared_observations >= gate["minimum_shared_stereo_identity_observations"]
        and all(per_side_pass))
    return {
        "purpose": "dual_camera_candidate_tracking_evaluation_against_shared_physical_ids",
        "reviewed_frames": len(reviewed),
        "total_frames": len(frames),
        "shared_stereo_identity_observations": shared_observations,
        "frames_with_at_least_3_shared_ids": int(sum(count >= 3 for count in shared_counts)),
        "master": master,
        "slave": slave,
        "acceptance_gate": gate,
        "accepted": accepted,
        "spin_solver_allowed": accepted,
        "reason": ("dual-view physical identity tracking passed" if accepted else
                   "dual-view physical identity evidence is incomplete or tracking failed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用双路共享物理 ID 真值评估候选检出率、保持率和换 ID 率")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("master_continuity_csv", type=Path)
    parser.add_argument("slave_continuity_csv", type=Path)
    parser.add_argument("--match-radius-px", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.annotations.with_name("dual_identity_evaluation.json")
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    report = evaluate(payload, load_candidates(args.master_continuity_csv),
                      load_candidates(args.slave_continuity_csv), args.match_radius_px)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
