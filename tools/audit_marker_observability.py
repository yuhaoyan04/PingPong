"""Audit spatial concentration and polarity balance in normalized marker packs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def audit_pack(annotation_path: Path) -> dict:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    values = []
    for record in payload["frames"]:
        image = cv2.imread(str(annotation_path.parent / record["frame"]))
        height, width = image.shape[:2]
        y, x = np.ogrid[:height, :width]
        normalized_radius = np.hypot(x - width / 2, y - height / 2) / (min(height, width) / 2)
        inner = normalized_radius < .60
        middle = (normalized_radius >= .60) & (normalized_radius < .80)
        rim = (normalized_radius >= .80) & (normalized_radius <= 1.0)
        off = image[..., 0] > 45
        on = image[..., 1] > 45
        active = off | on
        density = cv2.GaussianBlur(active.astype(np.float32), (0, 0), 5)
        inner_density = density[inner]
        off_count = int(np.count_nonzero(off & inner))
        on_count = int(np.count_nonzero(on & inner))
        polarity_balance = min(off_count, on_count) / max(1, off_count, on_count)
        values.append({
            "inner_active": int(np.count_nonzero(active & inner)),
            "middle_active": int(np.count_nonzero(active & middle)),
            "rim_active": int(np.count_nonzero(active & rim)),
            "inner_density_peak_to_mean": float(
                np.max(inner_density) / max(float(np.mean(inner_density)), 1e-9)),
            "inner_off": off_count,
            "inner_on": on_count,
            "inner_polarity_balance": polarity_balance,
        })
    result = {"frames": len(values)}
    for field in values[0]:
        samples = np.asarray([item[field] for item in values], np.float64)
        result[f"{field}_median"] = float(np.median(samples))
        result[f"{field}_p10"] = float(np.percentile(samples, 10))
        result[f"{field}_p90"] = float(np.percentile(samples, 90))
    gates = {
        "minimum_inner_active_median": 100,
        "minimum_inner_density_peak_to_mean_median": 6.0,
        "minimum_inner_polarity_balance_median": .25,
    }
    result["automatic_structure_gate"] = {
        "thresholds": gates,
        "passed": bool(
            result["inner_active_median"] >= gates["minimum_inner_active_median"]
            and result["inner_density_peak_to_mean_median"] >=
                gates["minimum_inner_density_peak_to_mean_median"]
            and result["inner_polarity_balance_median"] >=
                gates["minimum_inner_polarity_balance_median"]),
        "warning": "This gate measures event structure only; it cannot prove that physical marker centres or identities are human-labelable.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="审计归一化球内事件结构，不计算转速")
    parser.add_argument("--pack", action="append", required=True,
                        help="NAME=path/to/annotation_pack/annotations.json; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {}
    for specification in args.pack:
        if "=" not in specification:
            raise ValueError("--pack must be NAME=PATH")
        name, path = specification.split("=", 1)
        results[name] = audit_pack(Path(path))
    report = {
        "purpose": "normalized_ball_internal_event_structure_gate_not_spin_measurement",
        "packs": results,
        "required_next_gate": "human physical-marker centre and identity annotation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
