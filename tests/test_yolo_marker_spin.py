"""Physical synthetic motion, corruption and time/geometry boundary tests."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from pingpong_event_pipeline.marker_spin import fit_pairs, read_tracks, estimate_tracks, write_outputs


CFG = {"minimum_pairs": 8, "minimum_observations": 6, "minimum_duration_us": 2000,
       "hypotheses": 160, "window_us": 20000, "step_us": 10000,
       "max_gap_us": 750, "maximum_pair_us": 2000}


def synthetic_pairs(omega=(40., -35., 100.), dt=.0005, corrupt=False):
    points = np.array([[-.35, -.25], [.4, -.25], [-.3, .4], [.25, .3]])
    front = np.c_[points, -np.sqrt(1 - np.sum(points ** 2, axis=1))]
    pairs = []
    for marker, start in enumerate(front):
        run = []
        for frame in range(12):
            sphere = Rotation.from_rotvec(np.array(omega) * (frame * dt)).apply(start)
            if corrupt and marker == 0:
                angle = frame * .08
                sphere = np.array([.2 * np.cos(angle), .2 * np.sin(angle), -np.sqrt(.96)])
            assert sphere[2] < 0
            run.append({"sphere": sphere, "frame": frame, "radius": 256 / 2 / 1.2,
                        "timestamp": int(round(frame * dt * 1e6))})
        for i in range(len(run)):
            for lag in (1, 2, 4):
                if i + lag < len(run):
                    pairs.append((str(marker), run[i], run[i + lag], lag * dt))
    return pairs


def test_finite_motion_recovers_camera_angular_velocity_without_physical_promotion():
    fit = fit_pairs(synthetic_pairs(), CFG)
    assert fit["numerically_usable"]
    np.testing.assert_allclose(fit["omega_camera_rad_s"], [40, -35, 100], atol=1e-4)
    assert fit["candidate_spin_rpm"] == pytest.approx(np.linalg.norm([40, -35, 100]) * 60 / (2 * np.pi), rel=1e-6)
    assert fit["spin_rpm"] is None
    assert fit["accepted"] is False


def test_timestamp_scale_changes_spin_inversely():
    pairs = synthetic_pairs()
    a = fit_pairs(pairs, CFG)
    slower = [(segment, {**left, "timestamp": 2 * left["timestamp"]},
               {**right, "timestamp": 2 * right["timestamp"]}, 2 * dt) for segment, left, right, dt in pairs]
    b = fit_pairs(slower, CFG)
    assert b["candidate_spin_rpm"] == pytest.approx(a["candidate_spin_rpm"] / 2, rel=1e-6)


def test_bad_identity_is_rejected_by_robust_rotation_fit():
    fit = fit_pairs(synthetic_pairs(corrupt=True), CFG)
    assert fit["numerically_usable"]
    np.testing.assert_allclose(fit["omega_camera_rad_s"], [40, -35, 100], atol=1.)
    assert .70 <= fit["inlier_fraction"] < 1


def test_tiny_or_degenerate_track_is_not_reported_as_spin():
    fit = fit_pairs(synthetic_pairs()[:2], CFG)
    assert fit["candidate_spin_rpm"] is None
    pairs = synthetic_pairs()
    for _, a, b, _ in pairs:
        a["sphere"] = b["sphere"] = np.array([0., 0., -1.])
    fit = fit_pairs(pairs, CFG)
    assert not fit["numerically_usable"]


def input_fixture(tmp_path, source="detection_high", gap=500, scale=1.2):
    dataset = tmp_path / "data"
    folder = dataset / "unlabeled_inference/F3E/serve02_F3E"
    (folder / "binary").mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"camera": "F3E", "crop_radius_scale": scale}))
    manifest, tracks = [], []
    for frame in range(3):
        name = f"{frame}.png"
        Image.new("L", (256, 256)).save(folder / "binary" / name)
        manifest.append(dict(frame_index=frame, timestamp_us=10000 + frame * gap,
                             image=f"binary/{name}", track_support="ring_supported"))
        tracks.append(dict(camera="F3E", group="serve02_F3E", segment_id="S1", track_id=1,
                           segment_frame_offset=frame, segment_length=3, source=source,
                           center_x_px=160 + frame, center_y_px=130, **manifest[-1]))
    for path, rows in [(folder / "manifest.csv", manifest), (tmp_path / "tracks.csv", tracks)]:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return tmp_path / "tracks.csv", dataset


def test_crop_scale_and_original_microseconds_are_preserved(tmp_path):
    tracks, dataset = input_fixture(tmp_path)
    segments, sources, count = read_tracks(tracks, dataset)
    run = next(iter(segments.values()))
    assert count == 3
    assert sources["F3E/serve02_F3E"]["sphere_radius_px"] == pytest.approx(106.6666667)
    assert run[0]["sphere"][0] == pytest.approx(.3)
    assert run[1]["timestamp"] - run[0]["timestamp"] == 500
    report = estimate_tracks(tracks, dataset, CFG)
    assert report["usable_window_count"] == 0  # three frames qualify for export, not automatic spin
    write_outputs(report, tmp_path / "result")
    assert json.loads((tmp_path / "result/spin_report.json").read_text())["spin_rpm"] is None


@pytest.mark.parametrize("source,gap", [("prediction_hold", 500), ("detection_high", 1000), ("detection_high", 0)])
def test_prediction_only_or_discontinuous_segments_cannot_enter_solver(tmp_path, source, gap):
    tracks, dataset = input_fixture(tmp_path, source=source, gap=gap)
    with pytest.raises(ValueError):
        read_tracks(tracks, dataset)


def test_tampered_timestamps_fail_instead_of_using_video_fps(tmp_path):
    tracks, dataset = input_fixture(tmp_path)
    tracks.write_text(tracks.read_text().replace("10500", "10501"))
    with pytest.raises(ValueError, match="manifest mismatch"):
        read_tracks(tracks, dataset)


def test_fallback_frames_cannot_bridge_rotation_correspondences():
    from pingpong_event_pipeline.marker_spin import make_pairs
    run = [{"timestamp": i * 500, "track_support": "fallback_unverified" if i == 2 else "ring_supported"} for i in range(5)]
    pairs = make_pairs([("S1", run)], 0, 3000)
    assert len(pairs) == 2
    assert all(dt == .0005 for _, _, _, dt in pairs)
