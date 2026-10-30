"""Small local UI for reviewing a generated physical-dot annotation pack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="人工点击物理黑点中心")
    parser.add_argument("annotations", help="annotation_pack/annotations.json")
    args = parser.parse_args()
    manifest_path = Path(args.annotations)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    proposal_path = manifest_path.parent / "marker_region_proposals.json"
    proposal_lookup = {}
    if proposal_path.exists():
        proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        proposal_lookup = {record["frame"]: record["proposals"]
                           for record in proposal_payload.get("records", [])}
    index = 0
    show_hints = False
    window = "dot-centre annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def save() -> None:
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    def click(event, x, y, flags, userdata) -> None:
        labels = frames[index]["labels"]
        if event == cv2.EVENT_LBUTTONDOWN:
            labels.append([float(x), float(y)])
            frames[index]["status"] = "reviewed"
        elif event == cv2.EVENT_RBUTTONDOWN and labels:
            point = np.array([x, y], np.float64)
            nearest = int(np.argmin([np.linalg.norm(point - value) for value in labels]))
            labels.pop(nearest)
            frames[index]["status"] = "reviewed"

    cv2.setMouseCallback(window, click)
    while True:
        record = frames[index]
        image = cv2.imread(str(manifest_path.parent / record["frame"]))
        if show_hints:
            for point in proposal_lookup.get(record["frame"], []):
                center = (round(point["x"]), round(point["y"]))
                cv2.circle(image, center, 6, (255, 255, 0), 1)
        for number, point in enumerate(record["labels"], 1):
            center = tuple(np.rint(point).astype(int))
            cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(image, str(number), (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
        cv2.putText(image, f"{index+1}/{len(frames)} {record['status']}  click:add right:undo n/p x:unusable h:hints q",
                    (5, image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, image)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            save()
            break
        if key in (ord("n"), 13, 32):
            frames[index]["status"] = "reviewed"
            index = min(len(frames) - 1, index + 1)
            save()
        elif key in (ord("p"), 8):
            index = max(0, index - 1)
        elif key == ord("x"):
            record["labels"] = []
            record["status"] = "unusable"
            index = min(len(frames) - 1, index + 1)
            save()
        elif key == ord("u"):
            record["status"] = "unreviewed"
            save()
        elif key == ord("h"):
            show_hints = not show_hints
        elif key == ord("c"):
            record["labels"] = []
            record["status"] = "reviewed"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
