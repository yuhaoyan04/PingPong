"""Shared, validated annotation formats for marking and dataset export."""
from __future__ import annotations


def rectangle_to_yolo(shape: dict, width: int, height: int) -> str:
    """Convert a single reviewed Labelme marker rectangle to YOLO coordinates."""
    if shape.get('label') != 'marker' or shape.get('shape_type') != 'rectangle':
        raise ValueError('only rectangle labels named marker are supported')
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    points = shape.get('points', [])
    if len(points) != 2:
        raise ValueError('rectangle must contain two points')
    (x1, y1), (x2, y2) = points
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('rectangle is empty or outside the image')
    return f'0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}'
