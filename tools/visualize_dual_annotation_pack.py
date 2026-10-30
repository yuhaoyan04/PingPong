"""Render two synchronized marker-annotation packs side by side.

The red circles are annotation proposals only.  This tool deliberately assigns
no cross-camera or cross-frame marker identity and produces no spin estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_records(path: Path) -> Dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: Dict[int, dict] = {}
    for record in payload["frames"]:
        timestamp = int(record["timestamp_us"])
        if timestamp in records:
            raise ValueError(f"duplicate timestamp {timestamp} in {path}")
        records[timestamp] = record
    return records


def align_records(left: Dict[int, dict], right: Dict[int, dict]) -> Tuple[List[Tuple[int, dict, dict]], List[int], List[int]]:
    shared = sorted(set(left) & set(right))
    pairs = [(timestamp, left[timestamp], right[timestamp]) for timestamp in shared]
    return pairs, sorted(set(left) - set(right)), sorted(set(right) - set(left))


def load_proposals(path: Path | None) -> Dict[int, List[dict]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["timestamp_us"]): list(record["proposals"])
            for record in payload["records"]}


def draw_proposals(image: np.ndarray, proposals: List[dict]) -> np.ndarray:
    output = image.copy()
    for number, proposal in enumerate(proposals, 1):
        point = (round(float(proposal["x"])), round(float(proposal["y"])))
        cv2.circle(output, point, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(output, str(number), (point[0] + 5, point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 255, 255), 1, cv2.LINE_AA)
    return output


def distribution(values: List[int]) -> dict:
    if not values:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "minimum": 0, "maximum": 0}
    data = np.asarray(values, np.float64)
    return {
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
        "minimum": int(np.min(data)),
        "maximum": int(np.max(data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按绝对时间戳并排显示双路球面标记候选；不建立点身份、不计算转速")
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--master-proposals", type=Path)
    parser.add_argument("--slave-proposals", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    master = load_records(args.master_annotations)
    slave = load_records(args.slave_annotations)
    pairs, master_only, slave_only = align_records(master, slave)
    if not pairs:
        raise RuntimeError("the two annotation packs have no shared absolute timestamps")
    master_proposals = load_proposals(args.master_proposals)
    slave_proposals = load_proposals(args.slave_proposals)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_master = cv2.imread(str(args.master_annotations.parent / pairs[0][1]["frame"]))
    first_slave = cv2.imread(str(args.slave_annotations.parent / pairs[0][2]["frame"]))
    if first_master is None or first_slave is None:
        raise RuntimeError("could not read the first paired annotation images")
    height = max(first_master.shape[0], first_slave.shape[0])
    width = first_master.shape[1] + first_slave.shape[1]
    banner = 42
    video_path = args.output_dir / "dual_marker_review.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height + banner))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {video_path}")

    samples = []
    sample_indices = set(np.linspace(0, len(pairs) - 1, min(12, len(pairs)), dtype=int))
    master_counts: List[int] = []
    slave_counts: List[int] = []
    rows = []
    for index, (timestamp, master_record, slave_record) in enumerate(pairs):
        master_image = cv2.imread(str(args.master_annotations.parent / master_record["frame"]))
        slave_image = cv2.imread(str(args.slave_annotations.parent / slave_record["frame"]))
        if master_image is None or slave_image is None:
            raise RuntimeError(f"could not read paired images at {timestamp}")
        if master_image.shape[:2] != slave_image.shape[:2]:
            raise ValueError(f"paired image sizes differ at {timestamp}")
        mp = master_proposals.get(timestamp, [])
        sp = slave_proposals.get(timestamp, [])
        master_counts.append(len(mp))
        slave_counts.append(len(sp))
        master_image = draw_proposals(master_image, mp)
        slave_image = draw_proposals(slave_image, sp)
        cv2.putText(master_image, f"MASTER  candidates={len(mp)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(slave_image, f"SLAVE  candidates={len(sp)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((height + banner, width, 3), np.uint8)
        canvas[banner:banner + height] = np.hstack([master_image, slave_image])
        elapsed_ms = (timestamp - pairs[0][0]) / 1000.0
        cv2.putText(canvas, f"t={timestamp} us   elapsed={elapsed_ms:.3f} ms   exact timestamp pair",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "red = anonymous proposal; numbers are frame-local, not physical IDs",
                    (8, 35), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 220, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        if index in sample_indices:
            samples.append(canvas)
        rows.append({
            "timestamp_us": timestamp,
            "master_frame": master_record["frame"],
            "slave_frame": slave_record["frame"],
            "master_candidate_count": len(mp),
            "slave_candidate_count": len(sp),
        })
    writer.release()

    if samples:
        thumb_width = width // 2
        thumbs = [cv2.resize(item, (thumb_width, (height + banner) // 2),
                             interpolation=cv2.INTER_AREA) for item in samples]
        blank = np.zeros_like(thumbs[0])
        sheet_rows = []
        for begin in range(0, len(thumbs), 2):
            row = thumbs[begin:begin + 2]
            sheet_rows.append(np.hstack(row + [blank] * (2 - len(row))))
        cv2.imwrite(str(args.output_dir / "dual_marker_review_contact_sheet.jpg"),
                    np.vstack(sheet_rows))

    report = {
        "purpose": "dual_view_timestamp_and_anonymous_marker_review_no_spin",
        "status": "ready_for_human_physical_identity_annotation",
        "spin_solver_allowed": False,
        "paired_frames": len(pairs),
        "first_timestamp_us": pairs[0][0],
        "last_timestamp_us": pairs[-1][0],
        "timestamp_pairing": "exact_absolute_timestamp",
        "maximum_pair_skew_us": 0,
        "master_only_timestamps": master_only,
        "slave_only_timestamps": slave_only,
        "master_candidate_count": distribution(master_counts),
        "slave_candidate_count": distribution(slave_counts),
        "both_views_at_least_3_candidates_fraction": float(np.mean(
            (np.asarray(master_counts) >= 3) & (np.asarray(slave_counts) >= 3))),
        "video": str(video_path.resolve()),
        "contact_sheet": str((args.output_dir / "dual_marker_review_contact_sheet.jpg").resolve()),
        "frames": rows,
        "limitations": [
            "proposal numbers are local to each image and are not physical marker identities",
            "exact timestamps prove data alignment, not correct stereo marker correspondence",
            "human identity labels or a calibrated asymmetric marker template are required before spin",
        ],
    }
    (args.output_dir / "dual_marker_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
