"""Annotate shared physical marker IDs in synchronized dual-camera ball patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from pingpong_event_pipeline.stereo import StereoCalibration
from visualize_dual_annotation_pack import align_records, load_records


def color_for(identifier: int) -> tuple[int, int, int]:
    hue = (identifier * 47) % 180
    pixel = np.uint8([[[hue, 230, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def nearest_label(labels: List[dict], point: np.ndarray, maximum: float = 16.0):
    if not labels:
        return None
    distances = [np.linalg.norm(point - np.array([item["x"], item["y"]]))
                 for item in labels]
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= maximum else None


def next_identifier(frames: List[dict]) -> int:
    values = [item["id"] for frame in frames
              for side in ("master_labels", "slave_labels") for item in frame[side]]
    return max(values, default=0) + 1


def build_payload(master_path: Path, slave_path: Path) -> dict:
    master = load_records(master_path)
    slave = load_records(slave_path)
    pairs, master_only, slave_only = align_records(master, slave)
    if not pairs:
        raise ValueError("annotation packs have no exactly shared timestamps")
    master_meta = json.loads(master_path.read_text(encoding="utf-8"))
    slave_meta = json.loads(slave_path.read_text(encoding="utf-8"))
    if master_meta["image_size_px"] != slave_meta["image_size_px"]:
        raise ValueError("master and slave normalized patch sizes differ")
    return {
        "coordinate_system": "per_camera_normalized_patch_pixels",
        "image_size_px": master_meta["image_size_px"],
        "definition": "shared physical marker identities across cameras and consecutive frames",
        "master_annotations": str(master_path.resolve()),
        "slave_annotations": str(slave_path.resolve()),
        "master_only_timestamps": master_only,
        "slave_only_timestamps": slave_only,
        "controls": {
            "select": "click an existing ID in either previous/current view",
            "place": "click a current view; selection persists so the same ID can be placed in both cameras",
            "new": "press a to allocate a new ID, then click both current views",
            "delete": "right-click a label in a current view",
            "deselect": "press d",
            "propagate": "press g after placing shared seed IDs to propose them in later frames",
        },
        "frames": [{
            "timestamp_us": timestamp,
            "master_frame": master_record["frame"],
            "slave_frame": slave_record["frame"],
            "master_local_path_index": (int(master_record["local_path_index"])
                                         if "local_path_index" in master_record else None),
            "slave_local_path_index": (int(slave_record["local_path_index"])
                                        if "local_path_index" in slave_record else None),
            "master_labels": [],
            "slave_labels": [],
            "status": "unreviewed",
        } for timestamp, master_record, slave_record in pairs],
    }


def load_proposal_hints(annotation_path: Path) -> Dict[str, List[dict]]:
    path = annotation_path.parent / "marker_region_proposals.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {record["frame"]: record.get("proposals", [])
            for record in payload.get("records", [])}


def shared_ids(frame: dict) -> set[int]:
    master = {int(item["id"]) for item in frame["master_labels"]}
    slave = {int(item["id"]) for item in frame["slave_labels"]}
    return master & slave


def propagate_labels_optical_flow(previous_image: np.ndarray,
                                  current_image: np.ndarray,
                                  previous_labels: List[dict],
                                  proposals: List[dict],
                                  maximum_snap_px: float = 20.0,
                                  maximum_forward_backward_px: float = 4.0
                                  ) -> List[dict]:
    """Propagate seed IDs one frame and snap them to marker proposals.

    Returned labels remain proposals until a person explicitly reviews the
    frame, so automatic propagation cannot enter physical-template fitting.
    """
    if not previous_labels or not proposals:
        return []
    previous_gray = (cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
                     if previous_image.ndim == 3 else previous_image)
    current_gray = (cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                    if current_image.ndim == 3 else current_image)
    previous_gray = cv2.GaussianBlur(previous_gray, (3, 3), 0)
    current_gray = cv2.GaussianBlur(current_gray, (3, 3), 0)
    points = np.asarray([[item["x"], item["y"]]
                         for item in previous_labels], np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(25, 25), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             30, .01))
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, **options)
    if forward is None:
        return []
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, forward, None, **options)
    if backward is None:
        return []
    proposal_points = np.asarray([[item["x"], item["y"]]
                                  for item in proposals], np.float64)
    used = set(); propagated = []
    for index, label in enumerate(previous_labels):
        if not status_forward[index, 0] or not status_backward[index, 0]:
            continue
        fb_error = float(np.linalg.norm(backward[index, 0] - points[index, 0]))
        if fb_error > maximum_forward_backward_px:
            continue
        distances = np.linalg.norm(proposal_points - forward[index, 0], axis=1)
        order = np.argsort(distances)
        chosen = next((int(candidate) for candidate in order
                       if int(candidate) not in used), None)
        if chosen is None or distances[chosen] > maximum_snap_px:
            continue
        used.add(chosen)
        confidence = float(np.exp(-fb_error / max(maximum_forward_backward_px, 1e-6))
                           * np.exp(-distances[chosen] / max(maximum_snap_px, 1e-6)))
        propagated.append({
            "id": int(label["id"]),
            "x": float(proposal_points[chosen, 0]),
            "y": float(proposal_points[chosen, 1]),
            "source": "optical_flow_proposal",
            "confidence": confidence,
        })
    return propagated


def patch_to_full(point: np.ndarray, record: dict, image_size: int) -> np.ndarray:
    """Map a normalized ball-patch coordinate back to its source camera image."""
    radius = float(record["radius_px_full_image"])
    scale = radius / (0.5 * image_size)
    return np.asarray(point, np.float64) * scale + np.array([
        float(record["center_x_full_image"]) - radius,
        float(record["center_y_full_image"]) - radius,
    ])


def full_line_to_patch(line: np.ndarray, record: dict, image_size: int) -> np.ndarray:
    """Transform A*x+B*y+C=0 from full image to normalized patch pixels."""
    radius = float(record["radius_px_full_image"])
    scale = radius / (0.5 * image_size)
    offset = np.array([
        float(record["center_x_full_image"]) - radius,
        float(record["center_y_full_image"]) - radius,
    ])
    return np.array([line[0] * scale, line[1] * scale,
                     line[0] * offset[0] + line[1] * offset[1] + line[2]],
                    np.float64)


def epipolar_line_for_target(origin_point: np.ndarray, origin_record: dict,
                             target_record: dict, image_size: int,
                             calibration: StereoCalibration,
                             origin_is_master: bool) -> np.ndarray:
    full = patch_to_full(origin_point, origin_record, image_size)
    homogeneous = np.array([full[0], full[1], 1.0])
    line = (calibration.F @ homogeneous if origin_is_master
            else calibration.F.T @ homogeneous)
    return full_line_to_patch(line, target_record, image_size)


def line_distance(line: np.ndarray, point: np.ndarray) -> float:
    return float(abs(line @ np.array([point[0], point[1], 1.0])) /
                 max(np.hypot(line[0], line[1]), 1e-12))


def draw_epipolar_line(image: np.ndarray, line: np.ndarray) -> None:
    height, width = image.shape[:2]
    a, b, c = map(float, line)
    points = []
    if abs(b) > 1e-12:
        for x in (0, width - 1):
            y = -(a * x + c) / b
            if 0 <= y < height:
                points.append((x, round(y)))
    if abs(a) > 1e-12:
        for y in (0, height - 1):
            x = -(b * y + c) / a
            if 0 <= x < width:
                points.append((round(x), y))
    unique = list(dict.fromkeys(points))
    if len(unique) >= 2:
        cv2.line(image, unique[0], unique[1], (255, 180, 0), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="双路同步物理点 ID 标注；同一 ID 可放到主从相机当前帧")
    parser.add_argument("--master-annotations", type=Path, required=True)
    parser.add_argument("--slave-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration/stereo_calibration.npz"))
    args = parser.parse_args()

    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = build_payload(args.master_annotations, args.slave_annotations)
    payload.setdefault("controls", {})["propagate"] = (
        "press g after placing shared seed IDs to propose them in later frames")
    frames = payload["frames"]
    width, height = payload["image_size_px"]
    master_hints = load_proposal_hints(args.master_annotations)
    slave_hints = load_proposal_hints(args.slave_annotations)
    master_records = load_records(args.master_annotations)
    slave_records = load_records(args.slave_annotations)
    calibration = StereoCalibration(args.calibration)
    index = 0
    selected_id = None
    selected_source = None
    # The event images are sparse and the original 2x2 history view is hard to
    # read when assigning the first identities.  Start with the two current
    # views enlarged side by side; ``v`` toggles the history view when it is
    # useful for checking temporal continuity.
    show_hints = True
    focus_current = True
    window = "dual-camera physical marker identity"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1200, 700)

    def save() -> None:
        payload["summary"] = {
            "frames": len(frames),
            "reviewed_frames": sum(frame["status"] == "reviewed" for frame in frames),
            "unusable_frames": sum(frame["status"] == "unusable" for frame in frames),
            "shared_identity_observations": sum(len(shared_ids(frame)) for frame in frames),
            "frames_with_at_least_3_shared_ids": sum(len(shared_ids(frame)) >= 3 for frame in frames),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def propagate_forward_from_current() -> int:
        if len(shared_ids(frames[index])) < 3:
            return 0
        frames[index]["status"] = "reviewed"
        side_specs = (
            (args.master_annotations, master_hints, "master_labels", "master_frame"),
            (args.slave_annotations, slave_hints, "slave_labels", "slave_frame"),
        )
        proposed_frames = set()
        for annotations, hints, label_key, frame_key in side_specs:
            previous_image = cv2.imread(str(annotations.parent / frames[index][frame_key]))
            previous_labels = frames[index][label_key]
            for frame_index in range(index + 1, len(frames)):
                record = frames[frame_index]
                if record["status"] in ("reviewed", "unusable"):
                    previous_image = cv2.imread(str(
                        annotations.parent / record[frame_key]))
                    previous_labels = record[label_key]
                    continue
                current_image = cv2.imread(str(annotations.parent / record[frame_key]))
                labels = propagate_labels_optical_flow(
                    previous_image, current_image, previous_labels,
                    hints.get(record[frame_key], []))
                record[label_key] = labels
                if labels:
                    proposed_frames.add(frame_index)
                previous_image, previous_labels = current_image, labels
                if not labels:
                    break
        for frame_index in proposed_frames:
            if frames[frame_index]["status"] == "unreviewed":
                frames[frame_index]["status"] = "proposed"
        save()
        return len(proposed_frames)

    def click(event, x, y, flags, userdata) -> None:
        nonlocal selected_id, selected_source
        column, row = x // width, y // height
        if focus_current:
            if column not in (0, 1) or y >= height:
                return
            frame_index = index
            side = "master_labels" if column == 0 else "slave_labels"
            local = np.array([x % width, y], np.float64)
            current_column = True
        else:
            if column not in (0, 1) or row not in (0, 1):
                return
            frame_index = max(0, index - 1) if column == 0 else index
            side = "master_labels" if row == 0 else "slave_labels"
            local = np.array([x % width, y % height], np.float64)
            current_column = column == 1
        labels = frames[frame_index][side]
        nearest = nearest_label(labels, local)
        if event == cv2.EVENT_LBUTTONDOWN:
            if nearest is not None:
                selected_id = int(labels[nearest]["id"])
                selected_source = (frame_index, side, local.copy())
                return
            if not current_column:
                return
            if selected_id is None:
                selected_id = next_identifier(frames)
            labels[:] = [item for item in labels if int(item["id"]) != selected_id]
            labels.append({"id": int(selected_id), "x": float(local[0]), "y": float(local[1])})
            selected_source = (frame_index, side, local.copy())
            save()
        elif event == cv2.EVENT_RBUTTONDOWN and current_column and nearest is not None:
            labels.pop(nearest)
            save()

    cv2.setMouseCallback(window, click)
    save()
    while True:
        previous_index = max(0, index - 1)
        panels = []
        side_specs = (
            ("MASTER", args.master_annotations, master_hints, "master_labels", "master_frame"),
            ("SLAVE", args.slave_annotations, slave_hints, "slave_labels", "slave_frame"),
        )
        for row, (side_name, annotations, hints, label_key, frame_key) in enumerate(side_specs):
            row_panels = []
            frame_indices = (index,) if focus_current else (previous_index, index)
            for column, frame_index in enumerate(frame_indices):
                record = frames[frame_index]
                image = cv2.imread(str(annotations.parent / record[frame_key]))
                if image is None:
                    raise RuntimeError(f"could not read {annotations.parent / record[frame_key]}")
                if show_hints:
                    for hint_index, point in enumerate(hints.get(record[frame_key], []), 1):
                        cv2.circle(image, (round(point["x"]), round(point["y"])),
                                   7, (255, 80, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, f"P{hint_index}",
                                    (round(point["x"]) + 7, round(point["y"]) - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, .42,
                                    (255, 180, 255), 1, cv2.LINE_AA)
                is_current_panel = focus_current or column == 1
                if (is_current_panel and selected_source is not None
                        and selected_source[0] == index and selected_source[1] != label_key):
                    source_is_master = selected_source[1] == "master_labels"
                    timestamp = int(record["timestamp_us"])
                    source_record = (master_records[timestamp] if source_is_master
                                     else slave_records[timestamp])
                    target_record = (slave_records[timestamp] if source_is_master
                                     else master_records[timestamp])
                    line = epipolar_line_for_target(
                        selected_source[2], source_record, target_record, width,
                        calibration, source_is_master)
                    draw_epipolar_line(image, line)
                    for point in hints.get(record[frame_key], []):
                        xy = np.array([point["x"], point["y"]], np.float64)
                        if line_distance(line, xy) <= 5.0:
                            cv2.circle(image, (round(xy[0]), round(xy[1])),
                                       8, (255, 180, 0), 2, cv2.LINE_AA)
                for item in record[label_key]:
                    center = (round(item["x"]), round(item["y"]))
                    color = color_for(int(item["id"]))
                    thickness = 3 if selected_id == item["id"] else 1
                    cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 14, thickness)
                    cv2.putText(image, str(item["id"]), (center[0] + 5, center[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                title = (f"{side_name}  CURRENT" if focus_current else
                         f"{side_name}  {'PREVIOUS' if column == 0 else 'CURRENT'}")
                cv2.putText(image, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43,
                            (255, 255, 255), 1, cv2.LINE_AA)
                row_panels.append(image)
            panels.append(np.hstack(row_panels))
        canvas = np.hstack(panels) if focus_current else np.vstack(panels)
        shared = len(shared_ids(frames[index]))
        footer = np.zeros((42, canvas.shape[1], 3), np.uint8)
        cv2.putText(footer,
                    f"{index+1}/{len(frames)} status={frames[index]['status']} selected={selected_id} shared={shared}",
                    (7, 16), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(footer,
                    "a:new  n:review/next  p:previous  g:propagate  h:hints  v:view  right-click:delete  q:save",
                    (7, 35), cv2.FONT_HERSHEY_SIMPLEX, .36,
                    (210, 210, 210), 1, cv2.LINE_AA)
        canvas = np.vstack((canvas, footer))
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
            selected_source = None
            save()
        elif key in (ord("p"), 8):
            index = max(0, index - 1)
            selected_id = None
            selected_source = None
        elif key == ord("x"):
            frames[index]["master_labels"] = []
            frames[index]["slave_labels"] = []
            frames[index]["status"] = "unusable"
            index = min(len(frames) - 1, index + 1)
            selected_id = None
            selected_source = None
            save()
        elif key == ord("a"):
            selected_id = next_identifier(frames)
            selected_source = None
        elif key == ord("d"):
            selected_id = None
            selected_source = None
        elif key == ord("h"):
            show_hints = not show_hints
        elif key == ord("v"):
            focus_current = not focus_current
            selected_source = None
        elif key == ord("g"):
            propagate_forward_from_current()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
