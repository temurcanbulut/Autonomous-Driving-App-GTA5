"""Explicitly calibrated ground-plane approximation, with invalid cases rejected."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraCalibration:
    height_m: float
    vertical_fov_degrees: float
    horizon_y: float

    def distance(self, normalized_foot_y):
        if not (
            0.2 <= self.height_m <= 5
            and 20 <= self.vertical_fov_degrees <= 120
            and 0 < self.horizon_y < 1
        ):
            raise ValueError("Invalid camera calibration")
        if (
            not math.isfinite(normalized_foot_y)
            or normalized_foot_y <= self.horizon_y + 0.02
            or normalized_foot_y > 1
        ):
            return None
        focal_y = 0.5 / math.tan(math.radians(self.vertical_fov_degrees) / 2)
        return self.height_m * focal_y / (normalized_foot_y - self.horizon_y)


def corridor_conflict(box, lateral_velocity=0, horizon_seconds=0):
    x, y, width, height = box
    foot_y = y + height
    if foot_y < 0.45 or foot_y > 1.05:
        return False
    half_width = 0.07 + 0.29 * min(1, max(0, (foot_y - 0.45) / 0.55))
    projected_x = x + width / 2 + lateral_velocity * horizon_seconds
    return (
        projected_x + width / 2 >= 0.5 - half_width and projected_x - width / 2 <= 0.5 + half_width
    )
