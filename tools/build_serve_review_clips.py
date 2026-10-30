"""Split selected serve candidates and render raw/trajectory review videos."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.events import RawWindowReader
from pingpong_event_pipeline.models import Detection2D


def select_candidate(candidates, predicted, previous_radius, max_step=22.0,
                     radius_low=.72, radius_high=1.35):
    if predicted is None:
        return max(candidates, key=lambda value: value.score, default=None)
    plausible = [candidate for candidate in candidates
                 if np.linalg.norm(candidate.point - predicted) <= max_step
                 and radius_low * previous_radius <= candidate.radius <= radius_high * previous_radius]
    return max(
        plausible,
        key=lambda value: (value.score
                           - .012 * np.linalg.norm(value.point - predicted)
                           - .20 * abs(np.log(value.radius / previous_radius))),
        default=None,
    )


def robust_curve_fit(timestamps: np.ndarray, values: np.ndarray,
                     measured: np.ndarray, degree: int = 2) -> np.ndarray:
    """Fit a low-order flight curve while suppressing remaining bad detections."""
    t0 = float(np.mean(timestamps[measured]))
    scale = max(float(np.ptp(timestamps[measured])) * .5, 1.0)
    u = (timestamps.astype(np.float64) - t0) / scale
    design = np.vander(u[measured], degree + 1, increasing=True)
    weight = np.ones(np.count_nonzero(measured), np.float64)
    coefficients = np.zeros(degree + 1, np.float64)
    observed = np.asarray(values[measured], np.float64)
    for _ in range(8):
        root = np.sqrt(weight)
        coefficients = np.linalg.lstsq(design * root[:, None], observed * root,
                                        rcond=None)[0]
        residual = observed - design @ coefficients
        scale_r = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = np.minimum(1.0, 1.5 * scale_r / np.maximum(np.abs(residual), 1e-9))
    return np.vander(u, degree + 1, increasing=True) @ coefficients


def fit_global_rule_path(times: np.ndarray, candidates: list,
                         hypotheses: int = 3500, seed: int = 23):
    """Link independent circle+trail candidates with one smooth moving path.

    This deliberately avoids recursive prediction: a bad candidate in one bin
    cannot move the search gate in later bins.  RANSAC selects the longest
    contiguous candidate chain that agrees with a quadratic image trajectory.
    """
    usable = [i for i, values in enumerate(candidates) if values]
    if len(usable) < 8:
        return None
    origin = float(np.mean(times)); scale = max(float(np.ptp(times)) * .5, 1.0)
    u = (times.astype(np.float64) - origin) / scale
    design = np.column_stack((np.ones(len(u)), u, u * u))
    arrays = []
    for values in candidates:
        ranked = sorted(values, key=lambda item: item.score, reverse=True)[:12]
        arrays.append(np.array([[v.x, v.y, v.radius, v.score, v.event_count]
                                for v in ranked], np.float64))

    def evaluate(coefficients, threshold=15.0, maximum_gap=6):
        predicted = design @ coefficients
        nearest = np.full(len(times), -1, np.int32)
        distance = np.full(len(times), np.inf, np.float64)
        for i in usable:
            delta = arrays[i][:, :2] - predicted[i]
            j = int(np.argmin(np.sum(delta * delta, axis=1)))
            nearest[i] = j; distance[i] = np.linalg.norm(delta[j])
        inliers = np.flatnonzero(distance <= threshold)
        if len(inliers) < 8:
            return None
        groups = []; begin = 0
        for k in range(1, len(inliers)):
            if inliers[k] - inliers[k - 1] > maximum_gap + 1:
                groups.append(inliers[begin:k]); begin = k
        groups.append(inliers[begin:])
        group = max(groups, key=lambda g: (g[-1] - g[0], len(g)))
        if len(group) < 8:
            return None
        first, last = int(group[0]), int(group[-1])
        selected = group
        displacement = float(np.linalg.norm(predicted[last] - predicted[first]))
        if displacement < max(28.0, .45 * (last - first + 1)):
            return None
        velocity = (coefficients[1][None, :] + 2 * u[:, None] * coefficients[2][None, :]) / scale * 1e6
        speed = np.linalg.norm(velocity[first:last + 1], axis=1)
        if np.median(speed) < 500 or np.max(speed) > 30000:
            return None
        chosen_scores = np.array([arrays[i][nearest[i], 3] for i in selected])
        key = (last - first + 1, len(selected), -float(np.median(distance[selected])),
               float(np.sum(chosen_scores)))
        return key, first, last, nearest, distance, predicted

    rng = np.random.default_rng(seed)
    best = None; best_coefficients = None
    usable_array = np.asarray(usable)
    for _ in range(hypotheses):
        chosen_times = np.sort(rng.choice(usable_array, 3, replace=False))
        if chosen_times[-1] - chosen_times[0] < 25:
            continue
        points = []
        for index in chosen_times:
            values = arrays[int(index)]
            weights = np.maximum(values[:, 3], .03)
            j = int(rng.choice(len(values), p=weights / weights.sum()))
            points.append(values[j, :2])
        try:
            coefficients = np.linalg.solve(design[chosen_times], np.asarray(points))
        except np.linalg.LinAlgError:
            continue
        result = evaluate(coefficients)
        if result is not None and (best is None or result[0] > best[0]):
            best = result; best_coefficients = coefficients
    if best is None:
        return None

    # Refit to the selected observations and reselect a few times.
    for _ in range(4):
        result = evaluate(best_coefficients)
        if result is None:
            break
        _, first, last, nearest, distance, _ = result
        rows = [i for i in range(first, last + 1)
                if nearest[i] >= 0 and distance[i] <= 15.0]
        if len(rows) < 8:
            break
        points = np.array([arrays[i][nearest[i], :2] for i in rows])
        best_coefficients = np.linalg.lstsq(design[rows], points, rcond=None)[0]
    result = evaluate(best_coefficients)
    if result is None:
        return None
    _, first, last, nearest, distance, predicted = result
    chosen_index = np.full(len(times), -1, np.int32)
    for i in range(first, last + 1):
        if nearest[i] >= 0 and distance[i] <= 15.0:
            chosen_index[i] = nearest[i]

    def extend(direction: int) -> None:
        """Extend a globally anchored chain through bends and brief gaps."""
        nonlocal first, last
        rows = np.flatnonzero(chosen_index >= 0)
        if len(rows) < 3:
            return
        anchor = int(rows[-1] if direction > 0 else rows[0])
        history = rows[-6:] if direction > 0 else rows[:6][::-1]
        points = np.array([arrays[i][chosen_index[i], :2] for i in history])
        step = np.median(np.diff(points, axis=0), axis=0)
        position = points[-1].copy()
        last_measured_position = position.copy()
        radius = float(arrays[anchor][chosen_index[anchor], 2])
        misses = 0
        indexes = (range(anchor + 1, len(times)) if direction > 0
                   else range(anchor - 1, -1, -1))
        for i in indexes:
            expected = position + step
            values = arrays[i]
            if not len(values):
                position = expected; misses += 1
                if misses > 30:
                    break
                continue
            delta = values[:, :2] - expected
            distances = np.linalg.norm(delta, axis=1)
            normal_gate = min(38.0, max(16.0, 7.0 + 2.2 * np.linalg.norm(step)
                                        + 2.0 * misses))
            # During an impact the old velocity points to the wrong side of
            # the bounce.  Search around the last measured ball as a second
            # hypothesis, expanding through a short collision-event blackout.
            anchor_distances = np.linalg.norm(
                values[:, :2] - last_measured_position, axis=1)
            reacquire_gate = min(
                140.0, 28.0 + 6.0 * misses
                + 1.8 * np.linalg.norm(step) * (misses + 1))
            plausible = (((distances <= normal_gate)
                          | (anchor_distances <= reacquire_gate))
                         & (values[:, 2] >= .55 * radius)
                         & (values[:, 2] <= 1.75 * radius)
                         & (values[:, 3] >= .20))
            indexes_ok = np.flatnonzero(plausible)
            if not len(indexes_ok):
                position = expected; misses += 1
                if misses > 30:
                    break
                continue
            continuity_cost = np.minimum(
                distances[indexes_ok],
                .62 * anchor_distances[indexes_ok] + 2.0 * misses)
            costs = (continuity_cost - 7.0 * values[indexes_ok, 3]
                     + 2.5 * np.abs(np.log(values[indexes_ok, 2] / radius)))
            j = int(indexes_ok[int(np.argmin(costs))])
            point = values[j, :2]
            gap = misses + 1
            displacement = (point - last_measured_position) / gap
            # A direction reversal or a large acceleration is an impact, not
            # an outlier: reset velocity immediately instead of dragging the
            # next predictions along the pre-impact path.
            reversal = float(np.dot(displacement, step)) < 0.0
            acceleration = np.linalg.norm(displacement - step)
            if reversal or acceleration > max(6.0, 1.35 * np.linalg.norm(step)):
                step = displacement
            else:
                step = .68 * step + .32 * displacement
            position = point; radius = .8 * radius + .2 * values[j, 2]
            last_measured_position = point.copy()
            chosen_index[i] = j; misses = 0
            first = min(first, i); last = max(last, i)

    extend(1)
    extend(-1)
    measured = np.zeros(len(times), bool)
    scores = np.zeros(len(times), np.float64)
    supports = np.zeros(len(times), np.int32)
    radii_observed = np.full(len(times), np.nan, np.float64)
    centers_observed = np.full((len(times), 2), np.nan, np.float64)
    for i in range(first, last + 1):
        if chosen_index[i] >= 0:
            item = arrays[i][chosen_index[i]]
            measured[i] = True; radii_observed[i] = item[2]
            centers_observed[i] = item[:2]
            scores[i] = item[3]; supports[i] = round(item[4])
    if np.count_nonzero(measured) < 8:
        return None
    radii = np.exp(robust_curve_fit(
        times, np.log(np.maximum(np.nan_to_num(radii_observed, nan=20.0), 1.0)), measured))
    radii = np.clip(radii, 10.0, 75.0)
    # Interpolate only short missing bins, then smooth locally.  Unlike one
    # global polynomial this preserves a real image-plane bend or bounce.
    rows = np.flatnonzero(measured)
    centers = np.empty_like(centers_observed)
    for axis in range(2):
        centers[:, axis] = np.interp(np.arange(len(times)), rows,
                                     centers_observed[rows, axis])
        centers[:, axis] = cv2.GaussianBlur(
            centers[:, axis].reshape(-1, 1), (1, 0), 1.35).ravel()
    values = np.column_stack((centers, radii))
    return {
        "timestamps": times[first:last + 1], "values": values[first:last + 1],
        "scores": scores[first:last + 1], "supports": supports[first:last + 1],
        "guided_mask": np.zeros(last - first + 1, bool),
        "visibility_support": supports[first:last + 1],
        "visibility_threshold": 0,
    }


def refine_track_geometry(track: dict[str, np.ndarray], times: np.ndarray,
                          windows: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Refine the ball centre/radius after its motion path is known.

    Motion trails mainly extend behind the ball.  The extent perpendicular to
    velocity therefore estimates the true diameter more reliably; the leading
    edge then locates the centre along the velocity axis.
    """
    values = track["values"].copy()
    track_times = track["timestamps"]
    indexes = np.searchsorted(times, track_times)
    rows = np.arange(len(values))
    def median7(sequence: np.ndarray) -> np.ndarray:
        padded = np.pad(np.asarray(sequence, np.float64), (3, 3), mode="edge")
        return np.median(np.lib.stride_tricks.sliding_window_view(padded, 7), axis=1)

    # Re-evaluate once after moving the centre.  A single pass left a measured
    # 5--11 px residual on the close backspin/sidespin samples.
    for _ in range(2):
        shifts = np.full((len(values), 2), np.nan, np.float64)
        observed_radii = np.full(len(values), np.nan, np.float64)
        for row, (index, center, radius) in enumerate(
                zip(indexes, values[:, :2], values[:, 2])):
            if index >= len(windows) or times[index] != track_times[row]:
                continue
            current = windows[int(index)]
            points = np.column_stack((current["x"], current["y"])).astype(np.float64)
            relative = points - center
            gate = float(np.clip(max(62.0, 1.55 * float(radius)), 62.0, 78.0))
            inside = np.sum(relative * relative, axis=1) <= gate * gate
            local = relative[inside]
            polarity = current["polarity"][inside].astype(bool)
            if len(local) < 30 or min(np.count_nonzero(polarity),
                                      np.count_nonzero(~polarity)) < 5:
                continue
            # Warp every event from its own microsecond timestamp to the
            # current trajectory position.  This removes the asymmetric trail
            # exactly instead of approximating it with half a window of mean
            # velocity, which remained biased on the fastest clips.
            event_timestamps = current["timestamp"].astype(np.float64)[inside]
            past_x = np.interp(event_timestamps, track_times, values[:, 0])
            past_y = np.interp(event_timestamps, track_times, values[:, 1])
            compensated = np.column_stack((
                local[:, 0] + center[0] - past_x,
                local[:, 1] + center[1] - past_y,
            ))
            x_bounds = np.percentile(compensated[:, 0], (3.0, 97.0))
            y_bounds = np.percentile(compensated[:, 1], (3.0, 97.0))
            shift = np.array([np.mean(x_bounds), np.mean(y_bounds)], np.float64)
            refined_radius = .5 * max(float(np.ptp(x_bounds)),
                                      float(np.ptp(y_bounds)))
            if not 12.0 <= refined_radius <= 65.0:
                continue
            length = float(np.linalg.norm(shift))
            if length > 16.0:
                shift *= 16.0 / length
            shifts[row] = shift
            observed_radii[row] = refined_radius

        valid = np.isfinite(observed_radii)
        if np.count_nonzero(valid) < 8:
            break
        for axis in range(2):
            component = np.interp(rows, rows[valid], shifts[valid, axis])
            component = median7(component)
            component = cv2.GaussianBlur(
                component.reshape(-1, 1), (1, 0), 1.2).ravel()
            values[:, axis] += component
        radii = np.interp(rows, rows[valid], observed_radii[valid])
        radii = median7(radii)
        radii = cv2.GaussianBlur(radii.reshape(-1, 1), (1, 0), 1.2).ravel()
        median_radius = float(np.median(radii))
        values[:, 2] = np.clip(radii, .68 * median_radius, 1.30 * median_radius)
    refined = dict(track)
    refined["values"] = values
    return refined


def track_bidirectional(events: np.ndarray, times: np.ndarray, seed_timestamp: int,
                        roi: tuple[int, int, int, int]) -> dict[str, np.ndarray] | None:
    # In this fixed geometry the ball silhouette is far below 75 px radius.
    # Larger circles came from pairing the ball lobe with unrelated background
    # activity or a long trail, and are invalid for centre compensation.
    event_times = events["timestamp"].astype(np.int64, copy=False)

    candidate_cache = []
    window_cache = []

    def density_peak_candidates(current, timestamp):
        """Fallback candidates when only one ball lobe is cleanly visible."""
        x0, y0, x1, y1 = roi
        selected = current[(current["x"] >= x0) & (current["x"] < x1)
                           & (current["y"] >= y0) & (current["y"] < y1)]
        if len(selected) < 30:
            return []
        # Accumulate on a 4-pixel grid.  A full-resolution 780x720 Gaussian
        # for every 1 ms bin dominated runtime while adding no useful spatial
        # precision for a 20--60 px ball.
        bin_size = 4
        grid_w = (x1 - x0 + bin_size - 1) // bin_size
        grid_h = (y1 - y0 + bin_size - 1) // bin_size
        count = np.zeros((grid_h, grid_w), np.float32)
        xx = (selected["x"].astype(np.intp) - x0) // bin_size
        yy = (selected["y"].astype(np.intp) - y0) // bin_size
        np.add.at(count, (yy, xx), 1.0)
        smooth = cv2.GaussianBlur(count, (0, 0), 1.25)
        peak = float(smooth.max())
        if peak <= 0:
            return []
        local_max = cv2.dilate(smooth, np.ones((5, 5), np.uint8))
        locations = np.argwhere((smooth >= local_max - 1e-7)
                                & (smooth >= max(.025, .05 * peak)))
        ranked_all = sorted(
            locations, key=lambda q: float(smooth[q[0], q[1]]), reverse=True)
        ranked = list(ranked_all[:16])
        # A strong structure in the newly enabled left half must not evict the
        # ball candidates that made the original right-side corridor reliable.
        corridor_column = max(0, int((500 - x0) // bin_size))
        corridor_ranked = [q for q in ranked_all if int(q[1]) >= corridor_column][:16]
        seen = {(int(q[0]), int(q[1])) for q in ranked}
        ranked.extend(q for q in corridor_ranked
                      if (int(q[0]), int(q[1])) not in seen)
        candidates = []
        event_xy = np.column_stack((selected["x"], selected["y"])).astype(np.float64)
        for py, px in ranked:
            center0 = np.array([x0 + (px + .5) * bin_size,
                                y0 + (py + .5) * bin_size], np.float64)
            distance2 = np.sum((event_xy - center0) ** 2, axis=1)
            local = selected[distance2 <= 48.0 ** 2]
            if len(local) < 24:
                continue
            polarities = local["polarity"].astype(bool)
            if min(np.count_nonzero(polarities), np.count_nonzero(~polarities)) < 5:
                continue
            points = np.column_stack((local["x"], local["y"])).astype(np.float64)
            bounds_x = np.percentile(points[:, 0], (10, 90))
            bounds_y = np.percentile(points[:, 1], (10, 90))
            center = np.array([np.mean(bounds_x), np.mean(bounds_y)])
            radius = float(np.clip(.5 * max(np.ptp(bounds_x), np.ptp(bounds_y)), 10, 65))
            support = int(len(local))
            score = float(.18 + .25 * smooth[py, px] / peak
                          + .10 * min(np.count_nonzero(polarities), np.count_nonzero(~polarities))
                          / max(len(local), 1))
            candidates.append(Detection2D(int(timestamp), center[0], center[1],
                                          radius, score, support))
        candidates.sort(key=lambda value: value.score, reverse=True)
        return candidates

    def window_at(timestamp: int):
        # A long window turns a 5--8 px/ms ball into an elongated lobe and
        # shifts the enclosing circle backwards.  0.8 ms retains enough ball
        # events in these recordings while limiting motion bias.
        lo = np.searchsorted(event_times, timestamp - 800, side="left")
        hi = np.searchsorted(event_times, timestamp, side="right")
        return events[lo:hi]

    def guided_event_center(current, predicted, radius, gate):
        """Recover a centre from one dense event blob when lobe pairing fails."""
        if not len(current):
            return None
        search = min(95.0, max(34.0, float(radius) + float(gate)))
        dx = current["x"].astype(np.float64) - predicted[0]
        dy = current["y"].astype(np.float64) - predicted[1]
        nearby = current[dx * dx + dy * dy <= search * search]
        if len(nearby) < 20:
            return None
        bx = nearby["x"].astype(np.int64) // 4
        by = nearby["y"].astype(np.int64) // 4
        keys = by * 2048 + bx
        counts = np.bincount(keys)
        peak_key = int(np.argmax(counts))
        if counts[peak_key] < 3:
            return None
        peak = np.array([(peak_key % 2048) * 4 + 2,
                         (peak_key // 2048) * 4 + 2], np.float64)
        local_radius = max(14.0, .9 * float(radius))
        dd = (nearby["x"].astype(np.float64) - peak[0]) ** 2
        dd += (nearby["y"].astype(np.float64) - peak[1]) ** 2
        local = nearby[dd <= local_radius * local_radius]
        structured = []
        for polarity in (0, 1):
            selected = local[local["polarity"] == polarity]
            if not len(selected):
                structured.append(0); continue
            local_keys = ((selected["y"].astype(np.int64) // 4) * 2048
                          + selected["x"].astype(np.int64) // 4)
            local_counts = np.bincount(local_keys)
            structured.append(int(np.sum(local_counts[local_counts >= 3])))
        support = 2 * min(structured)
        if support < 40:
            return None
        center = np.array([np.median(local["x"]), np.median(local["y"])], np.float64)
        if np.linalg.norm(center - predicted) > gate:
            return None
        return center, support

    seed_index = int(np.argmin(np.abs(times - seed_timestamp)))
    for timestamp in times:
        current = window_at(int(timestamp))
        window_cache.append(current)
        # Generate proposals directly from dense bipolar blobs.  Running the
        # old all-pairs crescent matcher over the full 720-row corridor was
        # both quadratic in background components and prone to combining the
        # ball with its long trail.
        combined = density_peak_candidates(current, int(timestamp))
        combined.sort(key=lambda value: value.score, reverse=True)
        candidate_cache.append(combined)

    global_path = fit_global_rule_path(times, candidate_cache)
    # Also solve the historically useful right-side motion corridor.  Full
    # width is required when a ball exits left (topspin_c03), while the tighter
    # corridor can reject unrelated left-side activity and preserve a longer
    # curved pass (sidespin_c10).  Select by actual temporal coverage.
    corridor_candidates = [
        [candidate for candidate in row if candidate.x >= 500.0]
        for row in candidate_cache
    ]
    corridor_path = fit_global_rule_path(
        times, corridor_candidates, hypotheses=1400, seed=29)
    paths = [path for path in (global_path, corridor_path) if path is not None]
    if paths:
        best_path = max(paths, key=lambda path: (
            int(path["timestamps"][-1] - path["timestamps"][0]),
            int(np.count_nonzero(path["scores"] > 0)),
            float(np.sum(path["scores"]))))
        return refine_track_geometry(best_path, times, window_cache)

    def grow(candidate_seed_index, seed):
        values = np.full((len(times), 3), np.nan, np.float64)
        scores = np.zeros(len(times), np.float64)
        supports = np.zeros(len(times), np.int32)
        guided_mask = np.zeros(len(times), bool)
        values[candidate_seed_index] = [seed.x, seed.y, seed.radius]
        scores[candidate_seed_index] = seed.score
        supports[candidate_seed_index] = seed.event_count
        for direction in (1, -1):
            previous = values[candidate_seed_index].copy()
            velocity = np.zeros(2, np.float64)
            misses = 0
            indexes = (range(candidate_seed_index + 1, len(times)) if direction > 0
                       else range(candidate_seed_index - 1, -1, -1))
            for index in indexes:
                predicted = previous[:2] + direction * velocity
                # Brief weak-event intervals are common.  Coast through them,
                # then enlarge the gate gradually instead of terminating after
                # five misses.  Radius relaxation remains bounded so a large
                # background lobe cannot take over the track.
                gate = min(58.0, 22.0 + 3.0 * misses)
                relaxation = min(.18, .015 * misses)
                chosen = select_candidate(
                    candidate_cache[index], predicted, previous[2], gate,
                    max(.58, .72 - relaxation), min(1.58, 1.35 + relaxation))
                if chosen is None:
                    guided = guided_event_center(
                        window_cache[index], predicted, previous[2], gate)
                    if guided is not None:
                        center, support = guided
                        current = np.array([center[0], center[1], previous[2]], np.float64)
                        displacement = direction * (current[:2] - previous[:2])
                        velocity = .82 * velocity + .18 * displacement
                        values[index] = current; supports[index] = support
                        guided_mask[index] = True
                        previous = current; misses = max(0, misses - 2)
                        continue
                    misses += 1
                    values[index] = [predicted[0], predicted[1], previous[2]]
                    previous = values[index].copy()
                    continue
                current = np.array([chosen.x, chosen.y, chosen.radius], np.float64)
                displacement = direction * (current[:2] - previous[:2])
                velocity = .72 * velocity + .28 * displacement
                values[index] = current; scores[index] = chosen.score
                supports[index] = chosen.event_count
                previous = current; misses = 0
        return values, scores, supports, guided_mask

    # The activity peak is not guaranteed to be a valid ball frame in both
    # cameras.  Try candidates across the neighbourhood and retain the longest
    # coherent moving path rather than failing initialization at one timestamp.
    seed_order = sorted(range(len(times)), key=lambda i: abs(i - seed_index))
    proposals = []
    for index in seed_order[::2]:
        for seed in candidate_cache[index][:2]:
            proposals.append((index, seed))
        if len(proposals) >= 48:
            break
    best = None; best_key = None
    for candidate_seed_index, seed in proposals:
        values, scores, supports, guided_mask = grow(candidate_seed_index, seed)
        measured_indices = np.flatnonzero(scores > 0)
        if len(measured_indices) < 8:
            continue
        displacement = float(np.linalg.norm(
            values[measured_indices[-1], :2] - values[measured_indices[0], :2]))
        span = int(measured_indices[-1] - measured_indices[0] + 1)
        # Reject persistent static structures such as the net and table edge.
        if displacement < max(24.0, .45 * span):
            continue
        # Once motion and minimum evidence are satisfied, prefer the path that
        # covers the longest part of the pass.  Counting detections first could
        # select a dense middle fragment and discard valid entry/exit frames.
        key = (span, len(measured_indices), displacement, float(np.sum(scores)))
        if best_key is None or key > best_key:
            best_key = key; best = (values, scores, supports, guided_mask)
    if best is None:
        return None
    values, scores, supports, guided_mask = best

    measured = scores > 0
    if np.count_nonzero(measured) < 8:
        return None
    measured_first, measured_last = np.flatnonzero(measured)[[0, -1]]
    flight = np.column_stack((
        robust_curve_fit(times, values[:, 0], measured),
        robust_curve_fit(times, values[:, 1], measured),
        np.exp(robust_curve_fit(times, np.log(np.maximum(values[:, 2], 1.0)), measured)),
    ))
    flight[:, 2] = np.clip(flight[:, 2], 10.0, 75.0)
    extension = flight.copy()
    extension[guided_mask, :2] = values[guided_mask, :2]
    extension[guided_mask, 2] = values[guided_mask, 2]

    # Direct circle detection is allowed to disappear, but the ball still
    # produces dense events near the extrapolated flight path.  Use balanced
    # ON/OFF support to extend the visible interval until the ball truly leaves.
    visibility_support = np.zeros(len(times), np.int32)
    for index, timestamp in enumerate(times):
        lo = np.searchsorted(event_times, int(timestamp) - 800, side="left")
        hi = np.searchsorted(event_times, int(timestamp), side="right")
        current = events[lo:hi]
        if not len(current):
            continue
        radius = max(14.0, 1.35 * float(extension[index, 2]))
        dx = current["x"].astype(np.float64) - extension[index, 0]
        dy = current["y"].astype(np.float64) - extension[index, 1]
        near = dx * dx + dy * dy <= radius * radius
        if np.any(near):
            nearby = current[near]
            structured = []
            # A real ball creates repeated events in neighbouring 4x4 cells;
            # sensor noise is numerous but spatially scattered.  Counting only
            # cells with at least three events prevents background noise from
            # keeping an extrapolated circle alive after the ball has left.
            for polarity in (0, 1):
                selected = nearby[nearby["polarity"] == polarity]
                if not len(selected):
                    structured.append(0); continue
                keys = ((selected["y"].astype(np.int64) // 4) * 2048
                        + selected["x"].astype(np.int64) // 4)
                counts = np.bincount(keys)
                structured.append(int(np.sum(counts[counts >= 3])))
            visibility_support[index] = 2 * min(structured)
    reference = visibility_support[measured]
    positive_reference = reference[reference > 0]
    # Use weak measured frames as the visibility reference.  A median-based
    # threshold was hundreds of events on dense frames and incorrectly cut off
    # a still-visible partial ball near the sensor boundary.
    threshold = (max(40, int(.05 * np.percentile(positive_reference, 10)))
                 if len(positive_reference) else 40)
    visible = visibility_support >= threshold
    quiet_limit = 10
    first = measured_first; last_signal = measured_first; quiet = 0
    for index in range(measured_first - 1, -1, -1):
        if visible[index]:
            last_signal = index; quiet = 0
        else:
            quiet += 1
            if quiet > quiet_limit:
                break
    first = last_signal
    last = measured_last; last_signal = measured_last; quiet = 0
    for index in range(measured_last + 1, len(times)):
        if visible[index]:
            last_signal = index; quiet = 0
        else:
            quiet += 1
            if quiet > quiet_limit:
                break
    last = last_signal
    # Preserve at least the bounded short-horizon continuation used in V6;
    # event evidence may extend it farther but cannot shorten it abruptly.
    first = min(first, max(0, measured_first - 8))
    last = max(last, min(len(times) - 1, measured_last + 8))

    # Use direct/interpolated detections inside their measured span and the
    # robust physical flight model only for entry/exit extension.
    values[:measured_first] = extension[:measured_first]
    values[measured_last + 1:] = extension[measured_last + 1:]
    for axis in range(3):
        values[measured_first:measured_last + 1, axis] = np.interp(
            times[measured_first:measured_last + 1], times[measured], values[measured, axis])
        padded = np.pad(values[first:last + 1, axis], (2, 2), mode="edge")
        values[first:last + 1, axis] = np.convolve(padded, np.ones(5) / 5, mode="valid")
    return {"timestamps": times[first:last + 1], "values": values[first:last + 1],
            "scores": scores[first:last + 1], "supports": supports[first:last + 1],
            "guided_mask": guided_mask[first:last + 1],
            "visibility_support": visibility_support[first:last + 1],
            "visibility_threshold": threshold}


def event_panel(events: np.ndarray, event_times: np.ndarray, timestamp: int,
                window_us: int, sensor_size: tuple[int, int], panel_size: tuple[int, int],
                title: str, track: dict[str, np.ndarray] | None = None,
                compensate_motion: bool = False) -> np.ndarray:
    width, height = sensor_size; out_w, out_h = panel_size
    lo = np.searchsorted(event_times, timestamp - window_us, side="left")
    hi = np.searchsorted(event_times, timestamp, side="right")
    current = events[lo:hi]
    image = np.full((out_h, out_w, 3), 55, np.uint8)
    if len(current):
        event_x = current["x"].astype(np.float64)
        event_y = current["y"].astype(np.float64)
        if (compensate_motion and track is not None
                and track["timestamps"][0] <= timestamp <= track["timestamps"][-1]):
            times = track["timestamps"]; centers = track["values"][:, :2]
            current_times = event_times[lo:hi]
            target_x = np.interp(timestamp, times, centers[:, 0])
            target_y = np.interp(timestamp, times, centers[:, 1])
            event_x += target_x - np.interp(current_times, times, centers[:, 0])
            event_y += target_y - np.interp(current_times, times, centers[:, 1])
        x = np.clip(np.rint(event_x * out_w / width), 0, out_w - 1).astype(np.intp)
        y = np.clip(np.rint(event_y * out_h / height), 0, out_h - 1).astype(np.intp)
        p = current["polarity"].astype(bool)
        image[y[~p], x[~p]] = (255, 70, 30)
        image[y[p], x[p]] = (30, 240, 255)
        image = cv2.dilate(image, np.ones((2, 2), np.uint8))
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def add_track(image: np.ndarray, track: dict[str, np.ndarray] | None, timestamp: int,
              sensor_size: tuple[int, int]) -> None:
    if track is None:
        return
    times = track["timestamps"]
    if timestamp < times[0] or timestamp > times[-1]:
        return
    width, height = sensor_size; out_h, out_w = image.shape[:2]
    values = track["values"]
    cx = np.interp(timestamp, times, values[:, 0]) * out_w / width
    cy = np.interp(timestamp, times, values[:, 1]) * out_h / height
    radius = np.interp(timestamp, times, values[:, 2]) * .5 * (
        out_w / width + out_h / height)
    center = (round(cx), round(cy))
    nearest = int(np.argmin(np.abs(times - timestamp)))
    detected = bool(track["scores"][nearest] > 0)
    color = (0, 255, 0) if detected else (0, 165, 255)
    cv2.circle(image, center, max(2, round(radius)), color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    past = (times <= timestamp) & (times >= timestamp - 30000)
    points = values[past, :2].copy()
    points[:, 0] *= out_w / width; points[:, 1] *= out_h / height
    points = np.rint(points).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False,
                      (255, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成逐次发球原始事件和轨迹审核视频")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    # The old default stopped at y=240 on a 720-row sensor, so a perfectly
    # visible ball was discarded after crossing the upper third of the image.
    # Keep the useful right-side motion corridor but cover its full height.
    parser.add_argument("--roi", type=int, nargs=4, default=[0, 0, 1280, 720])
    parser.add_argument("--step-us", type=int, default=2000)
    parser.add_argument("--window-us", type=int, default=2000)
    parser.add_argument("--display-fps", type=float, default=30)
    args = parser.parse_args()

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    readers = [RawWindowReader(args.master_raw), RawWindowReader(args.slave_raw)]
    origin = max(reader.info.start_us for reader in readers)
    with Path(args.manifest).open(encoding="utf-8-sig") as handle:
        manifest = {row["clip_id"]: row for row in csv.DictReader(handle)}
    missing = set(args.clip_ids) - set(manifest)
    if missing:
        raise ValueError(f"missing clip IDs: {sorted(missing)}")
    selected = [manifest[clip_id] for clip_id in args.clip_ids]
    overall_start = origin + round(min(float(row["start_offset_s"]) for row in selected) * 1e6)
    overall_end = origin + round(max(float(row["end_offset_s"]) for row in selected) * 1e6)
    sequences = []
    for reader in readers:
        events = reader.window(overall_start - args.window_us, overall_end + 1)
        event_times = events["timestamp"].astype(np.int64, copy=False)
        if np.any(np.diff(event_times) < 0):
            events = events[np.argsort(event_times, kind="stable")]
            event_times = events["timestamp"].astype(np.int64, copy=False)
        sequences.append((events, event_times))

    summary_rows = []
    contact_tiles = []
    sensor_size = (readers[0].info.width, readers[0].info.height)
    panel_size = (640, round(640 * sensor_size[1] / sensor_size[0]))
    for row in selected:
        clip_id = row["clip_id"]
        clip_dir = output / clip_id; clip_dir.mkdir(exist_ok=True)
        start = origin + round(float(row["start_offset_s"]) * 1e6)
        end = origin + round(float(row["end_offset_s"]) * 1e6)
        peak = origin + round(float(row["peak_offset_s"]) * 1e6)
        track_times = np.arange(max(start, peak - 70000), min(end, peak + 160000) + 1,
                                1000, dtype=np.int64)
        tracks = []
        for camera, (events, event_times) in enumerate(sequences):
            lo = np.searchsorted(event_times, track_times[0] - 1500, side="left")
            hi = np.searchsorted(event_times, track_times[-1], side="right")
            track = track_bidirectional(events[lo:hi], track_times, peak, tuple(args.roi))
            tracks.append(track)
            if track is not None:
                np.savez(clip_dir / ("master_path.npz" if camera == 0 else "slave_path.npz"),
                         timestamps_us=track["timestamps"], centers_px=track["values"][:, :2],
                         radii_px=track["values"][:, 2], scores=track["scores"],
                         supports=track["supports"],
                         guided_mask=track["guided_mask"],
                         visibility_support=track["visibility_support"],
                         visibility_threshold=track["visibility_threshold"])

        video_size = (panel_size[0] * 2, panel_size[1])
        raw_path = clip_dir / "actual_events.mp4"
        tracked_path = clip_dir / "tracked_trajectory.mp4"
        raw_writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.display_fps, video_size)
        track_writer = cv2.VideoWriter(str(tracked_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       args.display_fps, video_size)
        render_times = np.arange(start, end + 1, args.step_us, dtype=np.int64)
        sampled = set(np.linspace(0, len(render_times) - 1, min(6, len(render_times)), dtype=int))
        for frame_index, timestamp in enumerate(render_times):
            raw_panels = [] ; tracked_panels = []
            for camera, title in enumerate(("MASTER actual events", "SLAVE actual events")):
                events, event_times = sequences[camera]
                panel = event_panel(events, event_times, int(timestamp), args.window_us,
                                    sensor_size, panel_size, title)
                raw_panels.append(panel)
                overlay = event_panel(
                    events, event_times, int(timestamp), min(args.window_us, 800),
                    sensor_size, panel_size,
                    title.replace("actual events", "0.8 ms motion-compensated"),
                    tracks[camera], True)
                add_track(overlay, tracks[camera], int(timestamp), sensor_size)
                tracked_panels.append(overlay)
            raw_frame = np.hstack(raw_panels); tracked_frame = np.hstack(tracked_panels)
            elapsed = (timestamp - start) / 1000
            cv2.putText(raw_frame, f"{clip_id}  {elapsed:.0f} ms", (510, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tracked_frame, f"{clip_id}  green=detected orange=predicted cyan=path", (330, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            raw_writer.write(raw_frame); track_writer.write(tracked_frame)
            if frame_index in sampled:
                contact_tiles.append(cv2.resize(tracked_frame, (640, panel_size[1] // 2),
                                                interpolation=cv2.INTER_AREA))
        raw_writer.release(); track_writer.release()
        camera_stats = []
        for track in tracks:
            camera_stats.append({
                "tracked": track is not None,
                "duration_ms": (float((track["timestamps"][-1] - track["timestamps"][0]) / 1000)
                                if track is not None else 0.0),
                "measured_frames": (int(np.count_nonzero(track["scores"] > 0))
                                    if track is not None else 0),
            })
        summary_rows.append({
            "clip_id": clip_id, "spin_class": row["run_id"],
            "start_offset_s": float(row["start_offset_s"]),
            "end_offset_s": float(row["end_offset_s"]),
            "joint_score": float(row["joint_score"]),
            "master": camera_stats[0], "slave": camera_stats[1],
            "actual_events_video": str(raw_path.resolve()),
            "tracked_trajectory_video": str(tracked_path.resolve()),
            "human_status": "unreviewed",
        })

    if contact_tiles:
        blank = np.zeros_like(contact_tiles[0]); rows = []
        for index in range(0, len(contact_tiles), 3):
            chunk = contact_tiles[index:index + 3]
            rows.append(np.hstack(chunk + [blank] * (3 - len(chunk))))
        cv2.imwrite(str(output / "trajectory_review_contact_sheet.jpg"), np.vstack(rows))
    report = {
        "purpose": "human_review_of_individual_real_serve_clips",
        "spin_class": selected[0]["run_id"],
        "clip_count": len(summary_rows),
        "clips": summary_rows,
        "acceptance": "keep clips where both cameras track the same complete ball pass",
        "warning": "Trajectory overlays are automatic candidates and require human review.",
    }
    (output / "serve_review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
