"""Short-horizon, image-space marker hypotheses; predictions are not observations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MarkerTrack:
    id: int
    x: float
    y: float
    vx: float = 0.0  # normalized patch pixels per update
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    status: str = 'observed'


class MarkerTracker:
    def __init__(self, gate_px=30.0, max_gap=3, min_hits=2, max_step=18.0):
        if gate_px <= 0 or max_gap < 0 or min_hits < 1 or max_step <= 0:
            raise ValueError('invalid tracker settings')
        self.gate_px, self.max_gap, self.min_hits, self.max_step = gate_px, max_gap, min_hits, max_step
        self.tracks = []
        self.next_id = 1

    def update(self, candidates):
        """Associate candidates and predict a bounded number of absent updates."""
        points = np.asarray([[p['x'], p['y']] for p in candidates], dtype=float).reshape(-1, 2)
        predictions = np.asarray([[s.x+s.vx, s.y+s.vy] for s in self.tracks], dtype=float).reshape(-1, 2)
        pairs = []
        if len(predictions) and len(points):
            d = np.linalg.norm(predictions[:, None, :] - points[None, :, :], axis=2)
            # Dummy columns make an unmatched track preferable to an infeasible association.
            padded = np.full((len(predictions), len(points) + len(predictions)), self.gate_px, dtype=float)
            padded[:, :len(points)] = np.where(d < self.gate_px, d, self.gate_px + 1e6)
            rows, cols = linear_sum_assignment(padded)
            pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if j < len(points) and d[i, j] < self.gate_px]
        assigned_tracks = {i for i, _ in pairs}
        assigned_candidates = {j for _, j in pairs}
        for i, j in pairs:
            s = self.tracks[i]
            # Displacement after misses is relative to the last prediction; smooth and bound velocity.
            delta = points[j] - np.array([s.x, s.y])
            velocity = .5*np.array([s.vx, s.vy]) + .5*delta
            norm = np.linalg.norm(velocity)
            if norm > self.max_step:
                velocity *= self.max_step / norm
            s.x, s.y = map(float, points[j])
            s.vx, s.vy = map(float, velocity)
            s.hits += 1; s.misses = 0; s.status = 'observed'
        alive = []
        for i, s in enumerate(self.tracks):
            if i not in assigned_tracks:
                s.misses += 1; s.status = 'predicted'
                s.x += s.vx; s.y += s.vy
            if s.misses <= self.max_gap:
                alive.append(s)
        self.tracks = alive
        for j, point in enumerate(points):
            if j not in assigned_candidates:
                self.tracks.append(MarkerTrack(self.next_id, float(point[0]), float(point[1])))
                self.next_id += 1
        return [s for s in self.tracks if s.hits >= self.min_hits]
