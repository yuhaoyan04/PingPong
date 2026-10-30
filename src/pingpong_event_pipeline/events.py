from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - binary/version incompatibility fallback
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


EVENT_DTYPE = np.dtype([
    ("x", "<u2"), ("y", "<u2"), ("polarity", "u1"), ("timestamp", "<u8")
])


@njit(cache=True)
def _ball_patch_kernel(x_values, y_values, polarities, timestamps, cx, cy, radius,
                       size, end_us, tau_us, vx, vy, max_age_us):
    last = np.full((2, size, size), -1, np.int64)
    scale = (0.5 * size) / max(radius, 1.0)
    for i in range(len(timestamps)):
        age = end_us - int(timestamps[i])
        if age < 0 or age > max_age_us:
            continue
        x = float(x_values[i]) + vx * age * 1e-6
        y = float(y_values[i]) + vy * age * 1e-6
        # The square normalized patch is only storage.  Events outside the
        # predicted opaque-ball disc are background and must never enter the
        # spin image after being translated with the ball velocity.
        if (x - cx) * (x - cx) + (y - cy) * (y - cy) > radius * radius:
            continue
        u = int(np.rint((x - cx) * scale + 0.5 * size))
        v = int(np.rint((y - cy) * scale + 0.5 * size))
        if 0 <= u < size and 0 <= v < size:
            p = int(polarities[i]) & 1
            if int(timestamps[i]) > last[p, v, u]:
                last[p, v, u] = int(timestamps[i])
    surface = np.zeros((2, size, size), np.float32)
    denominator = max(tau_us, 1.0)
    for p in range(2):
        for v in range(size):
            for u in range(size):
                if last[p, v, u] >= 0:
                    surface[p, v, u] = np.exp(-(end_us - last[p, v, u]) / denominator)
    return surface


@njit(cache=True)
def _ingest_timestamp_kernel(last, x_values, y_values, polarities, timestamps,
                             width, height):
    for i in range(len(timestamps)):
        x, y = int(x_values[i]), int(y_values[i])
        if 0 <= x < width and 0 <= y < height:
            p = int(polarities[i]) & 1
            timestamp = int(timestamps[i])
            if timestamp > last[p, y, x]:
                last[p, y, x] = timestamp


def sae(events: np.ndarray, width: int, height: int, end_us: int, tau_us: float) -> np.ndarray:
    """Return positive/negative surfaces of active events in [0, 1]."""
    surface = np.zeros((2, height, width), dtype=np.float32)
    if events.size == 0:
        return surface
    timestamps = events["timestamp"].astype(np.int64, copy=False)
    valid = (
        (events["x"] < width) & (events["y"] < height) &
        (timestamps <= end_us)
    )
    if not np.any(valid):
        return surface
    x = events["x"][valid].astype(np.intp)
    y = events["y"][valid].astype(np.intp)
    p = (events["polarity"][valid].astype(np.intp) & 1)
    t = timestamps[valid]
    last = np.full((2, height, width), -np.inf, dtype=np.float64)
    np.maximum.at(last, (p, y, x), t)
    active = np.isfinite(last)
    surface[active] = np.exp(-(end_us - last[active]) / max(float(tau_us), 1.0))
    return surface


def motion_compensated_sae(
    events: np.ndarray,
    width: int,
    height: int,
    end_us: int,
    tau_us: float,
    velocity_px_s: Tuple[float, float] = (0.0, 0.0),
    max_age_us: Optional[int] = None,
) -> np.ndarray:
    """Build an SAE after translating events to their predicted end positions.

    ``velocity_px_s`` is the image-plane velocity of the tracked ball.  For an
    event at time ``t``, its end-of-window location is ``p + v*(end-t)``.  This
    freezes translation of the ball boundary and markings, which otherwise
    become long trails at table-tennis speeds.  Rotation is deliberately left
    in the data because it is the signal used by the spin estimator.
    """
    surface = np.zeros((2, height, width), dtype=np.float32)
    if events.size == 0:
        return surface
    timestamps = events["timestamp"].astype(np.int64, copy=False)
    age_us = int(end_us) - timestamps
    valid = (timestamps <= end_us) & (age_us >= 0)
    if max_age_us is not None:
        valid &= age_us <= int(max_age_us)
    if not np.any(valid):
        return surface
    vx, vy = map(float, velocity_px_s)
    age_s = age_us[valid].astype(np.float64) * 1e-6
    x = np.rint(events["x"][valid].astype(np.float64) + vx * age_s).astype(np.int64)
    y = np.rint(events["y"][valid].astype(np.float64) + vy * age_s).astype(np.int64)
    p = events["polarity"][valid].astype(np.intp) & 1
    t = timestamps[valid]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(inside):
        return surface
    x, y, p, t = x[inside], y[inside], p[inside], t[inside]
    last = np.full((2, height, width), -np.inf, dtype=np.float64)
    np.maximum.at(last, (p, y, x), t)
    active = np.isfinite(last)
    surface[active] = np.exp(-(end_us - last[active]) / max(float(tau_us), 1.0))
    return surface


def motion_compensated_ball_patch(
    events: np.ndarray,
    center_px: Tuple[float, float],
    radius_px: float,
    size: int,
    end_us: int,
    tau_us: float,
    velocity_px_s: Tuple[float, float] = (0.0, 0.0),
    max_age_us: Optional[int] = None,
) -> np.ndarray:
    """Accumulate translated events directly into a normalized ball ROI."""
    if events.size == 0 or radius_px <= 0:
        return np.zeros((2, size, size), np.float32)
    vx, vy = map(float, velocity_px_s)
    age_limit = int(max_age_us if max_age_us is not None else np.iinfo(np.int64).max)
    if NUMBA_AVAILABLE:
        return _ball_patch_kernel(
            events["x"], events["y"], events["polarity"], events["timestamp"],
            float(center_px[0]), float(center_px[1]), float(radius_px), int(size),
            int(end_us), float(tau_us), vx, vy, age_limit)
    timestamps = events["timestamp"].astype(np.int64, copy=False)
    age_us = int(end_us) - timestamps
    valid = (age_us >= 0) & (age_us <= age_limit)
    if not np.any(valid):
        return np.zeros((2, size, size), np.float32)
    age_s = age_us[valid].astype(np.float64) * 1e-6
    x = events["x"][valid].astype(np.float64) + vx * age_s
    y = events["y"][valid].astype(np.float64) + vy * age_s
    on_ball = ((x - float(center_px[0])) ** 2 + (y - float(center_px[1])) ** 2
               <= float(radius_px) ** 2)
    if not np.any(on_ball):
        return np.zeros((2, size, size), np.float32)
    x, y = x[on_ball], y[on_ball]
    scale = (0.5 * size) / max(float(radius_px), 1.0)
    u = np.rint((x - float(center_px[0])) * scale + 0.5 * size).astype(np.int64)
    v = np.rint((y - float(center_px[1])) * scale + 0.5 * size).astype(np.int64)
    p = (events["polarity"][valid].astype(np.intp) & 1)[on_ball]
    t = timestamps[valid][on_ball]
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    last = np.full((2, size, size), -1, np.int64)
    np.maximum.at(last, (p[inside], v[inside], u[inside]), t[inside])
    surface = np.zeros((2, size, size), np.float32)
    active = last >= 0
    surface[active] = np.exp(-(end_us - last[active]) / max(float(tau_us), 1.0))
    return surface


def trajectory_compensated_ball_patch(
    events: np.ndarray,
    path_timestamps_us: np.ndarray,
    path_centers_px: np.ndarray,
    path_radii_px: np.ndarray,
    size: int,
    end_us: int,
    tau_us: float,
    max_age_us: Optional[int] = None,
) -> np.ndarray:
    """Normalize events with the complete time-varying ball trajectory.

    Each event is expressed relative to the interpolated ball centre and radius
    at *its own* timestamp.  This removes translation, acceleration and apparent
    scale change without removing rotation of surface markings.  It is more
    accurate than a constant-velocity warp for offline review of fast serves.
    """
    if events.size == 0:
        return np.zeros((2, size, size), np.float32)
    path_timestamps = np.asarray(path_timestamps_us, np.int64).reshape(-1)
    centers = np.asarray(path_centers_px, np.float64).reshape(-1, 2)
    radii = np.asarray(path_radii_px, np.float64).reshape(-1)
    if len(path_timestamps) < 2 or len(centers) != len(path_timestamps) or len(radii) != len(path_timestamps):
        raise ValueError("trajectory arrays must have equal length and at least two samples")
    timestamps = events["timestamp"].astype(np.int64, copy=False)
    age_us = int(end_us) - timestamps
    age_limit = int(max_age_us if max_age_us is not None else np.iinfo(np.int64).max)
    valid = (age_us >= 0) & (age_us <= age_limit)
    if not np.any(valid):
        return np.zeros((2, size, size), np.float32)
    timestamps = timestamps[valid]
    event_x = events["x"][valid].astype(np.float64)
    event_y = events["y"][valid].astype(np.float64)
    cx = np.interp(timestamps, path_timestamps, centers[:, 0])
    cy = np.interp(timestamps, path_timestamps, centers[:, 1])
    radius = np.interp(timestamps, path_timestamps, radii)
    usable = radius > 1.0
    if not np.any(usable):
        return np.zeros((2, size, size), np.float32)
    timestamps = timestamps[usable]
    radius = radius[usable]
    normalized_x = (event_x[usable] - cx[usable]) / radius
    normalized_y = (event_y[usable] - cy[usable]) / radius
    on_ball = normalized_x * normalized_x + normalized_y * normalized_y <= 1.0
    if not np.any(on_ball):
        return np.zeros((2, size, size), np.float32)
    half = 0.5 * float(size)
    u = np.rint(normalized_x[on_ball] * half + half).astype(np.int64)
    v = np.rint(normalized_y[on_ball] * half + half).astype(np.int64)
    p = (events["polarity"][valid][usable][on_ball].astype(np.intp) & 1)
    t = timestamps[on_ball]
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    last = np.full((2, size, size), -1, np.int64)
    np.maximum.at(last, (p[inside], v[inside], u[inside]), t[inside])
    surface = np.zeros((2, size, size), np.float32)
    active = last >= 0
    surface[active] = np.exp(-(int(end_us) - last[active]) / max(float(tau_us), 1.0))
    return surface


def render_bgr(surface: np.ndarray) -> np.ndarray:
    image = np.full((surface.shape[1], surface.shape[2], 3), 96, dtype=np.uint8)
    off = np.clip(surface[0] * 255, 0, 255).astype(np.uint8)
    on = np.clip(surface[1] * 255, 0, 255).astype(np.uint8)
    image[..., 0] = np.maximum(image[..., 0], off)
    image[..., 1] = np.maximum(image[..., 1], on)
    image[..., 2] = np.maximum(image[..., 2], on)
    return image


@dataclass
class RawInfo:
    path: Path
    start_us: int
    end_us: int
    width: int
    height: int


class RawWindowReader:
    def __init__(self, path: str | Path):
        from dvsense_driver.raw_file_reader import RawFileReader

        self.path = Path(path)
        self.reader = RawFileReader(str(self.path))
        if not self.reader.load_file():
            raise RuntimeError(f"无法加载 RAW: {self.path}")
        ok_start, start = self.reader.get_start_timestamp()
        ok_end, end = self.reader.get_end_timestamp()
        if not ok_start or not ok_end:
            raise RuntimeError(f"RAW 缺少时间范围: {self.path}")
        self.info = RawInfo(self.path, int(start), int(end), self.reader.get_width(), self.reader.get_height())

    def window(self, start_us: int, end_us: int) -> np.ndarray:
        # Some SDK seeks can return a different event set exactly at the seek
        # timestamp. Seek 1 ms earlier and apply our own half-open bounds.
        # This addresses an observed boundary discrepancy, not clock sync or
        # all possible decoder errors. The first file timestamp cannot be padded.
        lo, hi = int(start_us), int(end_us)
        if hi <= lo:
            return np.empty(0, dtype=EVENT_DTYPE)
        seek_start = max(int(self.info.start_us), lo - 1000)
        if hi <= seek_start:
            return np.empty(0, dtype=EVENT_DTYPE)
        events = self.reader.get_n_time_events(seek_start, hi - seek_start)
        if events is None or not len(events):
            return np.empty(0, dtype=EVENT_DTYPE)
        timestamp = events['timestamp']
        return events[(timestamp >= lo) & (timestamp < hi)]

    def seek(self, timestamp_us: int) -> None:
        self.reader.seek_time(int(timestamp_us))

    def next_duration(self, duration_us: int) -> np.ndarray:
        events = self.reader.get_n_time_events(int(duration_us))
        return events if events is not None else np.empty(0, dtype=EVENT_DTYPE)


class IncrementalTimeSurface:
    """Persistent last-event timestamps; each incoming event is ingested once."""

    def __init__(self, width: int, height: int):
        self.width, self.height = int(width), int(height)
        self.last = np.full((2, height, width), -1, np.int64)

    def ingest(self, events: np.ndarray) -> None:
        if events.size == 0:
            return
        x = events["x"].astype(np.intp); y = events["y"].astype(np.intp)
        p = events["polarity"].astype(np.intp) & 1
        valid = (x < self.width) & (y < self.height)
        np.maximum.at(self.last, (p[valid], y[valid], x[valid]),
                      events["timestamp"][valid].astype(np.int64))

    def surface(self, end_us: int, tau_us: float, max_age_us: int) -> np.ndarray:
        age = int(end_us) - self.last
        valid = (self.last >= 0) & (age >= 0) & (age <= int(max_age_us))
        result = np.zeros(self.last.shape, np.float32)
        result[valid] = np.exp(-age[valid] / max(float(tau_us), 1.0))
        return result

    def region(self, bounds, end_us: int, tau_us: float, max_age_us: int) -> np.ndarray:
        x0, y0, x1, y1 = map(int, bounds)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.width, x1), min(self.height, y1)
        sampled = self.last[:, y0:y1, x0:x1]
        age = int(end_us) - sampled
        valid = (sampled >= 0) & (age >= 0) & (age <= int(max_age_us))
        result = np.zeros(sampled.shape, np.float32)
        result[valid] = np.exp(-age[valid] / max(float(tau_us), 1.0))
        return result

    def patch(self, center_px, radius_px: float, size: int, end_us: int,
              tau_us: float, max_age_us: int) -> np.ndarray:
        half = 0.5 * int(size)
        coordinates = (np.arange(size, dtype=np.float64) - half) * float(radius_px) / half
        x = np.rint(float(center_px[0]) + coordinates).astype(np.int64)
        y = np.rint(float(center_px[1]) + coordinates).astype(np.int64)
        valid_x = (x >= 0) & (x < self.width); valid_y = (y >= 0) & (y < self.height)
        x = np.clip(x, 0, self.width - 1); y = np.clip(y, 0, self.height - 1)
        sampled = self.last[:, y[:, None], x[None, :]]
        age = int(end_us) - sampled
        valid = ((sampled >= 0) & (age >= 0) & (age <= int(max_age_us)) &
                 valid_y[None, :, None] & valid_x[None, None, :])
        result = np.zeros((2, size, size), np.float32)
        result[valid] = np.exp(-age[valid] / max(float(tau_us), 1.0))
        return result


def synchronized_batches(
    left: RawWindowReader,
    right: RawWindowReader,
    step_us: int,
    left_correction_us: int = 0,
    right_correction_us: int = 0,
    range_start_us: Optional[int] = None,
    range_end_us: Optional[int] = None,
    warmup_us: int = 0,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield each event once in consecutive synchronized batches."""
    common_start = max(left.info.start_us + left_correction_us,
                       right.info.start_us + right_correction_us)
    output_start = max(common_start, int(range_start_us) if range_start_us is not None else common_start)
    cursor = max(common_start, output_start - int(warmup_us))
    end = min(left.info.end_us + left_correction_us, right.info.end_us + right_correction_us)
    if range_end_us is not None:
        end = min(end, int(range_end_us))
    left.seek(cursor - left_correction_us); right.seek(cursor - right_correction_us)
    while cursor + step_us <= end:
        left_events = left.next_duration(step_us); right_events = right.next_duration(step_us)
        cursor += step_us
        if left_correction_us and left_events.size:
            left_events = left_events.copy(); left_events["timestamp"] += left_correction_us
        if right_correction_us and right_events.size:
            right_events = right_events.copy(); right_events["timestamp"] += right_correction_us
        if cursor >= output_start:
            yield cursor, left_events, right_events


def synchronized_windows(
    left: RawWindowReader,
    right: RawWindowReader,
    window_us: int,
    step_us: int,
    left_correction_us: int = 0,
    right_correction_us: int = 0,
    range_start_us: Optional[int] = None,
    range_end_us: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    start = max(left.info.start_us + left_correction_us, right.info.start_us + right_correction_us)
    end = min(left.info.end_us + left_correction_us, right.info.end_us + right_correction_us)
    if range_start_us is not None:
        start = max(start, int(range_start_us))
    if range_end_us is not None:
        end = min(end, int(range_end_us))
    cursor = start
    left.seek(start - left_correction_us)
    right.seek(start - right_correction_us)
    left_buffer = left.next_duration(window_us)
    right_buffer = right.next_duration(window_us)
    first = True
    while cursor + window_us <= end:
        if not first:
            left_buffer = np.concatenate((left_buffer, left.next_duration(step_us)))
            right_buffer = np.concatenate((right_buffer, right.next_duration(step_us)))
        first = False
        left_start, right_start = cursor - left_correction_us, cursor - right_correction_us
        left_stop, right_stop = left_start + window_us, right_start + window_us
        left_buffer = left_buffer[(left_buffer["timestamp"] >= left_start) & (left_buffer["timestamp"] <= left_stop)]
        right_buffer = right_buffer[(right_buffer["timestamp"] >= right_start) & (right_buffer["timestamp"] <= right_stop)]
        # 输出统一时钟域，后续 SAE 和双目窗口不需要知道各自修正量。
        left_events, right_events = left_buffer, right_buffer
        if left_correction_us and left_events.size:
            left_events = left_events.copy(); left_events["timestamp"] += left_correction_us
        if right_correction_us and right_events.size:
            right_events = right_events.copy(); right_events["timestamp"] += right_correction_us
        yield cursor + window_us, left_events, right_events
        cursor += step_us
