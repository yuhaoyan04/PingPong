"""Polarity-agnostic event clusters for exploratory marker review, not truth labels."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_events(image, neighbor_radius=4, min_neighbors=2, merge_radius=6,
                   min_pixels=3, maximum=6, max_extent=None, max_aspect=None):
    if min(neighbor_radius, merge_radius) < 0 or min_neighbors < 1 or min_pixels < 1 or maximum < 1:
        raise ValueError('invalid cluster settings')
    active = np.max(image, axis=2) > 45
    h, w = active.shape
    yy, xx = np.ogrid[:h, :w]
    # Only the physical ball mask; no additional silhouette exclusion.
    active &= (xx - w / 2)**2 + (yy - h / 2)**2 <= (min(w, h) / 2)**2
    kernel = np.ones((2 * neighbor_radius + 1, 2 * neighbor_radius + 1), np.uint8)
    count = cv2.filter2D(active.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    kept = active & (count >= min_neighbors)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_radius + 1, 2 * merge_radius + 1))
    expanded = cv2.dilate(kept.astype(np.uint8), k)
    n, labels = cv2.connectedComponents(expanded, connectivity=8)
    components = []
    for label in range(1, n):
        ys, xs = np.where(kept & (labels == label))
        if len(xs) >= min_pixels:
            width, height = int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1
            if max_extent is not None and max(width, height) > max_extent:
                continue
            if max_aspect is not None and max(width, height) / min(width, height) > max_aspect:
                continue
            components.append({'x': float(xs.mean()), 'y': float(ys.mean()), 'pixels': int(len(xs)),
                               'label': label, 'width': width, 'height': height})
    components.sort(key=lambda c: c['pixels'], reverse=True)
    components = components[:maximum]
    denoised = np.zeros_like(image)
    denoised[kept] = image[kept]
    return components, denoised, labels, kept


def confirm_offline(frames, distance=20.0, size_ratio=4.0):
    """Confirm a cluster with a similar nearby cluster in an adjacent time window.

    Future confirmation recovers the first observation without inventing events.
    Returns one set of confirmed proposal indexes per input frame.
    """
    if distance <= 0 or size_ratio < 1:
        raise ValueError('invalid offline confirmation settings')
    accepted = []
    for index, frame in enumerate(frames):
        confirmed = set()
        neighbors = []
        if index > 0:
            neighbors.extend(frames[index - 1])
        if index + 1 < len(frames):
            neighbors.extend(frames[index + 1])
        for j, proposal in enumerate(frame):
            for other in neighbors:
                delta = (proposal['x'] - other['x'])**2 + (proposal['y'] - other['y'])**2
                ratio = max(proposal['pixels'], other['pixels']) / min(proposal['pixels'], other['pixels'])
                if delta <= distance**2 and ratio <= size_ratio:
                    confirmed.add(j)
                    break
        accepted.append(confirmed)
    return accepted


def associate_offline(frames, distance=20.0, size_ratio=6.0, max_gap=1,
                      min_observations=3, min_strong=1):
    """Associate weak/strong clusters one-to-one, then retrospectively qualify tracks.

    A gap is only a matching allowance: no synthetic observation is returned.
    Every input proposal must have x, y, pixels and strong fields.
    """
    if distance <= 0 or size_ratio < 1 or max_gap < 0 or min_observations < 2 or min_strong < 1:
        raise ValueError('invalid association settings')
    tracks = []
    for frame_index, frame in enumerate(frames):
        active = [track for track in tracks if frame_index - track['last_frame'] <= max_gap + 1]
        cost = np.full((len(active), len(frame) + len(active)), distance, dtype=float)
        for i, track in enumerate(active):
            dt = frame_index - track['last_frame']
            predicted = np.array(track['position']) + dt * np.array(track['velocity'])
            for j, proposal in enumerate(frame):
                ratio = max(track['pixels'], proposal['pixels']) / min(track['pixels'], proposal['pixels'])
                d = np.linalg.norm(predicted - [proposal['x'], proposal['y']])
                if d < distance and ratio <= size_ratio:
                    cost[i, j] = d
                else:
                    cost[i, j] = distance + 1e6
        pairs = []
        if active and frame:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(i), int(j)) for i, j in zip(rows, cols)
                     if j < len(frame) and cost[i, j] < distance]
        used = set()
        for i, j in pairs:
            track = active[i]
            proposal = frame[j]
            dt = frame_index - track['last_frame']
            position = np.array([proposal['x'], proposal['y']])
            displacement = (position - np.array(track['position'])) / dt
            velocity = .5 * np.array(track['velocity']) + .5 * displacement
            speed = np.linalg.norm(velocity)
            if speed > distance / 2:
                velocity *= (distance / 2) / speed
            track.update(last_frame=frame_index, position=position.tolist(), velocity=velocity.tolist(),
                         pixels=proposal['pixels'])
            track['members'].append((frame_index, j))
            track['strong'] += int(proposal['strong'])
            used.add(j)
        for j, proposal in enumerate(frame):
            if j not in used:
                tracks.append({'last_frame': frame_index, 'position': [proposal['x'], proposal['y']],
                               'velocity': [0., 0.], 'pixels': proposal['pixels'],
                               'members': [(frame_index, j)], 'strong': int(proposal['strong'])})
    accepted = [set() for _ in frames]
    for track in tracks:
        if len(track['members']) >= min_observations and track['strong'] >= min_strong:
            for frame_index, j in track['members']:
                accepted[frame_index].add(j)
    return accepted
