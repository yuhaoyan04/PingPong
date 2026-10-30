"""Conservative spatial denoising for review, not marker classification."""
import cv2
import numpy as np


def neighbor_filter(binary, radius=3, min_neighbors=2):
    """Keep occupied pixels with nearby occupied pixels of either polarity.

    Counts exclude the pixel itself. No morphology is applied, so the result
    is always a subset of the original and the two polarities may support each
    other. Sparse genuine features can be removed; always review the original.
    """
    if radius < 1 or min_neighbors < 0:
        raise ValueError('invalid neighborhood parameters')
    mask = np.asarray(binary) != 0
    if mask.ndim != 2:
        raise ValueError('expected a 2D binary image')
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.float32)
    kernel[radius, radius] = 0
    count = cv2.filter2D(mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (mask & (count >= min_neighbors)).astype(np.uint8) * 255


def temporal_neighbor_filter(current, previous, radius=3, min_neighbors=2, motion_radius=6):
    """Experimental: spatially supported current pixels near prior-window activity.

    Both inputs must already be motion-compensated into the same ball coordinates.
    A newly visible or quickly rotating marker may be removed, so this is for
    comparison, not a ground-truth marker mask.
    """
    if motion_radius < 0:
        raise ValueError('motion_radius must be nonnegative')
    spatial = neighbor_filter(current, radius, min_neighbors)
    old = np.asarray(previous) != 0
    if old.shape != spatial.shape:
        raise ValueError('frames must have identical shapes')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*motion_radius+1, 2*motion_radius+1))
    nearby = cv2.dilate(old.astype(np.uint8), kernel) != 0
    return ((spatial != 0) & nearby).astype(np.uint8)*255


def multiscale_contrast_filter(binary, threshold=0.035, small_sigma=3., large_sigma=12.):
    """Keep original occupied pixels at locally prominent activity concentrations.

    The score is the difference between Gaussian-smoothed occupancy at two
    scales, not marker probability. Output is strictly a subset of the input.
    Dense trails and ball rims can also score highly, while faint marks can be
    lost. Parameters are in crop pixels and must be checked per camera/scale.
    """
    if threshold < 0 or small_sigma <= 0 or large_sigma <= small_sigma:
        raise ValueError('invalid contrast filter parameters')
    mask = np.asarray(binary) != 0
    if mask.ndim != 2:
        raise ValueError('expected a 2D binary image')
    occupied = mask.astype(np.float32)
    local = cv2.GaussianBlur(occupied, (0, 0), small_sigma)
    background = cv2.GaussianBlur(occupied, (0, 0), large_sigma)
    return (mask & ((local-background) >= threshold)).astype(np.uint8)*255
