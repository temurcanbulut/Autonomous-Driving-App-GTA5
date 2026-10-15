"""Class-aware short-lived tracks and image-space expansion estimates."""

import math
import statistics
from collections import deque
from dataclasses import dataclass, field


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1e-9, aw * ah + bw * bh - intersection)


@dataclass
class Track:
    identifier: int
    label: str
    box: tuple
    confidence: float
    timestamp: float
    history: deque = field(default_factory=lambda: deque(maxlen=16))
    states: deque = field(default_factory=lambda: deque(maxlen=16))

    def lateral_velocity(self):
        if len(self.states) < 2:
            return 0.0
        slopes = []
        recent = [s for s in self.states if self.states[-1][0] - s[0] <= 0.25]
        for i, (t0, b0) in enumerate(recent):
            for t1, b1 in recent[i + 1 :]:
                if 0.025 <= t1 - t0 <= 0.25:
                    slopes.append(((b1[0] + b1[2] / 2) - (b0[0] + b0[2] / 2)) / (t1 - t0))
        return max(-2.0, min(2.0, statistics.median(slopes))) if slopes else 0.0

    def predicted_box(self, timestamp):
        x, y, width, height = self.box
        return (
            x + self.lateral_velocity() * min(0.3, max(0, timestamp - self.timestamp)),
            y,
            width,
            height,
        )

    def optical_ttc(self):
        if len(self.history) < 3:
            return None
        first, last = self.history[0], self.history[-1]
        dt = last[0] - first[0]
        if dt < 0.05:
            return None
        growth = (math.sqrt(last[1]) - math.sqrt(first[1])) / dt
        if growth <= 1e-4:
            return None
        return math.sqrt(last[1]) / growth


class Tracker:
    def __init__(self, max_age=0.3):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.last_time = None

    def update(self, detections, timestamp):
        if not math.isfinite(timestamp):
            self.tracks.clear()
            raise ValueError("Finite track timestamp required")
        if self.last_time is not None and timestamp <= self.last_time:
            self.tracks.clear()
        self.last_time = timestamp
        self.tracks = {
            key: track
            for key, track in self.tracks.items()
            if 0 <= timestamp - track.timestamp <= self.max_age
        }
        available = set(self.tracks)
        predicted_boxes = {
            key: track.predicted_box(timestamp) for key, track in self.tracks.items()
        }
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            box = detection.get("box", ())
            if (
                len(box) != 4
                or not all(math.isfinite(x) for x in box)
                or box[2] <= 0
                or box[3] <= 0
                or box[0] < 0
                or box[1] < 0
                or box[0] + box[2] > 1.000001
                or box[1] + box[3] > 1.000001
            ):
                continue
            if not math.isfinite(detection["confidence"]) or not 0 <= detection["confidence"] <= 1:
                continue
            candidates = [
                (
                    max(
                        iou(self.tracks[key].box, detection["box"]),
                        iou(predicted_boxes[key], detection["box"]),
                    ),
                    key,
                )
                for key in available
                if self.tracks[key].label == detection["label"]
            ]
            overlap, matched = max(candidates, default=(0, None))
            if overlap < 0.15:
                matched = self.next_id
                self.next_id += 1
                self.tracks[matched] = Track(
                    matched,
                    detection["label"],
                    detection["box"],
                    detection["confidence"],
                    timestamp,
                )
            else:
                available.remove(matched)
            track = self.tracks[matched]
            track.box, track.confidence, track.timestamp = (
                detection["box"],
                detection["confidence"],
                timestamp,
            )
            track.history.append((timestamp, max(1e-8, detection["box"][2] * detection["box"][3])))
            track.states.append((timestamp, tuple(detection["box"])))
        # Missed tracks are kept for association, but must not act as fresh hazards.
        return [track for track in self.tracks.values() if track.timestamp == timestamp]
