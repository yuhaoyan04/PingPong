"""Measure short-term continuity of anonymous marker-region proposals.

Track numbers are diagnostic associations in normalized image space.  They are
not verified physical marker identities and must not be used to estimate spin.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from pingpong_event_pipeline.events import RawWindowReader, motion_compensated_ball_patch
from propose_marker_regions import propose
from visualize_ball_observability import colored_patch


@dataclass
class Track:
    identifier: int
    position: np.ndarray
    velocity: np.ndarray
    density: float
    last_frame: int
    missed: int = 0
    observations: list[dict] = field(default_factory=list)

    def predicted(self, frame_index: int) -> np.ndarray:
        return self.position + self.velocity * (frame_index - self.last_frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="量化匿名标记候选的短时连续性；结果不是物理点身份或转速")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--window-us", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--last-index", type=int,
                        help="inclusive; defaults to the final path index")
    parser.add_argument("--gate-px", type=float, default=16.0)
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument("--ambiguity-margin-px", type=float, default=3.0)
    parser.add_argument("--inner-radius-fraction", type=float, default=.40,
                        help="relative to square size; .40 is 80%% of ball radius")
    parser.add_argument("--display-fps", type=float, default=25.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.path)
    all_timestamps = loaded["timestamps_us"].astype(np.int64)
    all_centers = loaded["centers_px"].astype(np.float64)
    all_radii = loaded["radii_px"].astype(np.float64)
    dt = np.gradient(all_timestamps).astype(np.float64) * 1e-6
    all_center_velocity = np.column_stack([
        np.gradient(all_centers[:, axis]) / dt for axis in range(2)
    ])
    first = max(0, args.first_index)
    last = min(len(all_timestamps) - 1,
               args.last_index if args.last_index is not None else len(all_timestamps) - 1)
    if last < first:
        raise ValueError("last-index must be greater than or equal to first-index")
    selected = slice(first, last + 1)
    timestamps = all_timestamps[selected]
    centers = all_centers[selected]
    radii = all_radii[selected]
    center_velocity = all_center_velocity[selected]
    reader = RawWindowReader(args.raw)
    events = reader.window(int(timestamps[0] - args.window_us), int(timestamps[-1] + 1))
    event_times = events["timestamp"].astype(np.int64, copy=False)
    if np.any(np.diff(event_times) < 0):
        events = events[np.argsort(event_times, kind="stable")]
        event_times = events["timestamp"].astype(np.int64, copy=False)

    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    next_identifier = 1
    rows = []
    frames = []
    ambiguous_assignments = 0
    matched_assignments = 0
    proposal_counts = []
    for frame_index, timestamp in enumerate(timestamps):
        lo = np.searchsorted(event_times, timestamp - args.window_us, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        current = events[lo:hi]
        age_s = (timestamp - current["timestamp"].astype(np.int64)) * 1e-6
        compensated_x = current["x"].astype(np.float64) + center_velocity[frame_index, 0] * age_s
        compensated_y = current["y"].astype(np.float64) + center_velocity[frame_index, 1] * age_s
        inside = ((compensated_x - centers[frame_index, 0]) ** 2
                  + (compensated_y - centers[frame_index, 1]) ** 2
                  <= radii[frame_index] ** 2)
        patch = motion_compensated_ball_patch(
            current[inside], centers[frame_index], radii[frame_index], args.patch_size,
            int(timestamp), max(100, args.window_us / 2), center_velocity[frame_index],
            args.window_us)
        image = colored_patch(patch)
        candidates = propose(image, sigma=5.0, percentile=90.0, nms_px=25,
                             inner_radius_fraction=args.inner_radius_fraction, maximum=16)
        proposal_counts.append(len(candidates))
        candidate_xy = np.asarray([[item["x"], item["y"]] for item in candidates],
                                  np.float64).reshape(-1, 2)
        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()
        assignment_details: dict[int, tuple[float, bool]] = {}
        if active_tracks and len(candidate_xy):
            predictions = np.asarray([track.predicted(frame_index) for track in active_tracks])
            distances = np.linalg.norm(predictions[:, None, :] - candidate_xy[None, :, :], axis=2)
            density_cost = np.asarray([
                [2.0 * abs(np.log((candidate["density"] + 1e-5) /
                                  (track.density + 1e-5))) for candidate in candidates]
                for track in active_tracks
            ])
            costs = distances + density_cost
            track_indices, candidate_indices = linear_sum_assignment(costs)
            for track_index, candidate_index in zip(track_indices, candidate_indices):
                distance = float(distances[track_index, candidate_index])
                if distance > args.gate_px:
                    continue
                track = active_tracks[track_index]
                candidate = candidates[candidate_index]
                elapsed = max(1, frame_index - track.last_frame)
                displacement = candidate_xy[candidate_index] - track.position
                track.velocity = .65 * track.velocity + .35 * displacement / elapsed
                track.position = candidate_xy[candidate_index]
                track.density = .7 * track.density + .3 * float(candidate["density"])
                track.last_frame = frame_index
                track.missed = 0
                alternatives = np.sort(distances[track_index])
                margin = (float(alternatives[1] - alternatives[0])
                          if len(alternatives) > 1 else float("inf"))
                ambiguous = margin < args.ambiguity_margin_px
                observation = {
                    "frame_index": frame_index,
                    "timestamp_us": int(timestamp),
                    "x": float(track.position[0]),
                    "y": float(track.position[1]),
                    "density": float(candidate["density"]),
                    "assignment_distance_px": distance,
                    "assignment_margin_px": margin,
                    "ambiguous": ambiguous,
                }
                track.observations.append(observation)
                assigned_tracks.add(track_index)
                assigned_candidates.add(candidate_index)
                assignment_details[candidate_index] = (distance, ambiguous)
                matched_assignments += 1
                ambiguous_assignments += int(ambiguous)

        still_active = []
        for track_index, track in enumerate(active_tracks):
            if track_index not in assigned_tracks:
                track.missed += 1
            if track.missed <= args.max_gap_frames:
                still_active.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = still_active
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in assigned_candidates:
                continue
            track = Track(
                next_identifier,
                np.array([candidate["x"], candidate["y"]], np.float64),
                np.zeros(2, np.float64),
                float(candidate["density"]),
                frame_index,
            )
            track.observations.append({
                "frame_index": frame_index,
                "timestamp_us": int(timestamp),
                "x": float(candidate["x"]),
                "y": float(candidate["y"]),
                "density": float(candidate["density"]),
                "assignment_distance_px": None,
                "assignment_margin_px": None,
                "ambiguous": False,
            })
            active_tracks.append(track)
            next_identifier += 1

        for track in active_tracks:
            if track.last_frame != frame_index:
                continue
            observation = track.observations[-1]
            rows.append({
                "frame_index": frame_index,
                "timestamp_us": int(timestamp),
                "track_id": track.identifier,
                "x": observation["x"],
                "y": observation["y"],
                "density": observation["density"],
                "assignment_distance_px": observation["assignment_distance_px"],
                "assignment_margin_px": observation["assignment_margin_px"],
                "ambiguous": observation["ambiguous"],
            })
            center = tuple(np.rint(track.position).astype(int))
            color = (0, 0, 255) if observation["ambiguous"] else (0, 255, 0)
            cv2.circle(image, center, 5, color, 1)
            cv2.putText(image, str(track.identifier), (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 255, 255), 1)
        cv2.putText(image,
                    f"{frame_index}  green=associated red=ambiguous  diagnostic IDs only",
                    (4, args.patch_size - 8), cv2.FONT_HERSHEY_SIMPLEX, .38,
                    (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(image)
    finished_tracks.extend(active_tracks)

    video_path = output / "marker_continuity.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.display_fps, (args.patch_size, args.patch_size))
    for frame in frames:
        writer.write(frame)
    writer.release()
    sample_indices = np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int)
    samples = [frames[index] for index in sample_indices]
    if samples:
        columns = 5
        blank = np.zeros_like(samples[0])
        sheet_rows = [np.hstack(samples[i:i + columns]
                                + [blank] * (columns - len(samples[i:i + columns])))
                      for i in range(0, len(samples), columns)]
        cv2.imwrite(str(output / "marker_continuity_contact_sheet.jpg"),
                    np.vstack(sheet_rows))
    with (output / "marker_continuity.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = list(rows[0])
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    track_records = []
    for track in finished_tracks:
        if not track.observations:
            continue
        span_frames = (track.observations[-1]["frame_index"]
                       - track.observations[0]["frame_index"] + 1)
        track_records.append({
            "track_id": track.identifier,
            "observations": len(track.observations),
            "span_frames": span_frames,
            "start_frame": track.observations[0]["frame_index"],
            "end_frame": track.observations[-1]["frame_index"],
            "ambiguous_observations": int(sum(item["ambiguous"]
                                               for item in track.observations)),
        })
    lengths = np.asarray([item["observations"] for item in track_records], np.int64)
    report = {
        "purpose": "short_term_anonymous_proposal_continuity_not_physical_dot_identity_or_spin",
        "frames": len(timestamps),
        "source_first_index": first,
        "source_last_index": last,
        "frame_step_us_median": float(np.median(np.diff(timestamps))),
        "window_us": args.window_us,
        "association_gate_px": args.gate_px,
        "max_gap_frames": args.max_gap_frames,
        "inner_radius_fraction_of_image": args.inner_radius_fraction,
        "proposals_per_frame_median": float(np.median(proposal_counts)),
        "tracks_created": len(track_records),
        "track_observations_median": float(np.median(lengths)),
        "track_observations_p90": float(np.percentile(lengths, 90)),
        "tracks_at_least_5_observations": int(np.count_nonzero(lengths >= 5)),
        "tracks_at_least_10_observations": int(np.count_nonzero(lengths >= 10)),
        "tracks_at_least_20_observations": int(np.count_nonzero(lengths >= 20)),
        "matched_assignments": matched_assignments,
        "ambiguous_assignments": ambiguous_assignments,
        "ambiguous_assignment_fraction": (ambiguous_assignments / matched_assignments
                                          if matched_assignments else None),
        "track_records": track_records,
        "video": str(video_path.resolve()),
        "contact_sheet": str((output / "marker_continuity_contact_sheet.jpg").resolve()),
        "csv": str((output / "marker_continuity.csv").resolve()),
        "limitations": [
            "track IDs associate anonymous density regions, not verified physical markers",
            "identity switches cannot be measured before manual labels exist",
            "the conservative inner mask excludes markers near the projected limb",
            "clip-specific occlusion, path error, and field-of-view clipping are not modelled",
        ],
    }
    (output / "marker_continuity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "track_records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
