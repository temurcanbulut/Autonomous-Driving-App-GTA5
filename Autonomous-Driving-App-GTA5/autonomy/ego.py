"""Explicit camera-specific ego-vehicle polygon; never inferred from detections."""

import math

import cv2
import numpy as np

VEHICLE_IDS = frozenset({13, 14, 15, 16, 17})


def validate_polygon(points):
    if not isinstance(points, list) or not 3 <= len(points) <= 32:
        raise ValueError("Ego polygon needs 3..32 normalized points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Invalid ego polygon point")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in point
        ):
            raise ValueError("Ego polygon coordinates must be finite within [0,1]")
    polygon = np.asarray(points, dtype=np.float32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0:
        raise ValueError("Ego polygon must be convex with positive area")
    return polygon


def polygon_mask(points, width, height):
    polygon = validate_polygon(points)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid mask dimensions")
    output = np.zeros((height, width), dtype=np.uint8)
    pixels = np.rint(polygon * np.array([width - 1, height - 1])).astype(np.int32)
    cv2.fillConvexPoly(output, pixels, 1)
    return output.astype(bool)
