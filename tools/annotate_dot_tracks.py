"""Annotate persistent physical marker IDs in consecutive normalized ball frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def color_for(identifier: int) -> tuple[int, int, int]:
    hue = (identifier * 47) % 180
    pixel = np.uint8([[[hue, 230, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="连续帧物理黑点身份标注：先选前帧 ID，再点击当前帧对应位置")
    parser.add_argument("annotations", help="consecutive annotation_pack/annotations.json")
    parser.add_argument("--output", help="defaults to track_annotations.json beside input")
    args = parser.parse_args()
    source_path = Path(args.annotations)
    output_path = Path(args.output) if args.output else source_path.parent / "track_annotations.json"
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        payload = {
            "coordinate_system": source["coordinate_system"],
            "image_size_px": source["image_size_px"],
            "definition": "persistent physical marker identities across consecutive frames",
            "frames": [dict(frame, identity_labels=[], status="unreviewed")
                       for frame in source["frames"]],
        }
    frames = payload["frames"]
    index = 0
    selected_id: int | None = None
    proposal_path = source_path.parent / "marker_region_proposals.json"
    proposals_by_frame = {}
    if proposal_path.exists():
        proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        proposals_by_frame = {
            record["frame"]: record.get("proposals", [])
            for record in proposal_payload.get("records", [])
        }
    show_hints = False
    window = "physical-dot identity annotation"
    width, height = payload["image_size_px"]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def save() -> None:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    def nearest_label(labels, point, maximum=16.0):
        if not labels:
            return None
        distances = [np.linalg.norm(point - np.array([item["x"], item["y"]]))
                     for item in labels]
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] <= maximum else None

    def next_identifier() -> int:
        values = [item["id"] for frame in frames for item in frame["identity_labels"]]
        return max(values, default=0) + 1

    def click(event, x, y, flags, userdata) -> None:
        nonlocal selected_id
        panel = x // width
        local = np.array([x % width, y], np.float64)
        if event == cv2.EVENT_LBUTTONDOWN and panel == 0 and index > 0:
            labels = frames[index - 1]["identity_labels"]
            nearest = nearest_label(labels, local)
            if nearest is not None:
                selected_id = int(labels[nearest]["id"])
        elif event == cv2.EVENT_LBUTTONDOWN and panel == 1:
            identifier = selected_id if selected_id is not None else next_identifier()
            labels = frames[index]["identity_labels"]
            labels[:] = [item for item in labels if item["id"] != identifier]
            labels.append({"id": int(identifier), "x": float(local[0]), "y": float(local[1])})
            frames[index]["status"] = "reviewed"
            selected_id = None
            save()
        elif event == cv2.EVENT_RBUTTONDOWN and panel == 1:
            labels = frames[index]["identity_labels"]
            nearest = nearest_label(labels, local)
            if nearest is not None:
                labels.pop(nearest)
                frames[index]["status"] = "reviewed"
                save()

    cv2.setMouseCallback(window, click)
    # Create the destination immediately and persist every edit. Closing an
    # OpenCV window with its title-bar X must not discard an annotation session.
    save()
    while True:
        panel_indices = [max(0, index - 1), index, min(len(frames) - 1, index + 1)]
        panels = []
        for panel_number, frame_index in enumerate(panel_indices):
            record = frames[frame_index]
            image = cv2.imread(str(source_path.parent / record["frame"]))
            if show_hints:
                for point in proposals_by_frame.get(record["frame"], []):
                    cv2.circle(image, (round(point["x"]), round(point["y"])),
                               5, (180, 80, 180), 1)
            for item in record["identity_labels"]:
                center = (round(item["x"]), round(item["y"]))
                color = color_for(int(item["id"]))
                thickness = 3 if selected_id == item["id"] else 1
                cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 14, thickness)
                cv2.putText(image, str(item["id"]), (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            title = ("PREVIOUS: click ID" if panel_number == 0 else
                     "CURRENT: place ID" if panel_number == 1 else "NEXT: context")
            cv2.putText(image, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
        canvas = np.hstack(panels)
        cv2.putText(canvas,
                    f"{index+1}/{len(frames)} status={frames[index]['status']} selected={selected_id}  n/p x:unusable a:new c:clear h:hints q:save",
                    (5, height - 7), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            save()
            break
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            save()
            break
        if key in (ord("n"), 13, 32):
            frames[index]["status"] = "reviewed"
            index = min(len(frames) - 1, index + 1)
            selected_id = None
            save()
        elif key in (ord("p"), 8):
            index = max(0, index - 1)
            selected_id = None
        elif key == ord("x"):
            frames[index]["identity_labels"] = []
            frames[index]["status"] = "unusable"
            index = min(len(frames) - 1, index + 1)
            selected_id = None
            save()
        elif key == ord("a"):
            selected_id = None
        elif key == ord("h"):
            show_hints = not show_hints
        elif key == ord("c"):
            frames[index]["identity_labels"] = []
            frames[index]["status"] = "reviewed"
            selected_id = None
            save()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
