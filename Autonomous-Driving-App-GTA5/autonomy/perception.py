"""Trained semantic perception. Runtime and evaluation share preprocessing."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import ARCHITECTURES, build_model

ACTORS = {
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}
EXPECTED_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def validate_metadata(metadata):
    if (
        metadata.get("model") not in ARCHITECTURES
        or tuple(metadata.get("classes", ())) != EXPECTED_CLASSES
    ):
        raise ValueError("Unsupported checkpoint architecture or class mapping")
    if not (128 <= metadata.get("width", 0) <= 1920 and 128 <= metadata.get("height", 0) <= 1080):
        raise ValueError("Invalid checkpoint input dimensions")


def components(mask, confidence, minimum_area=16, minimum_confidence=0.65):
    height, width = mask.shape
    detections = []
    for class_id, label in ACTORS.items():
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            x, y, box_width, box_height, area = stats[component]
            if area < minimum_area:
                continue
            selected = labels[y : y + box_height, x : x + box_width] == component
            certainty = float(confidence[y : y + box_height, x : x + box_width][selected].mean())
            if certainty < minimum_confidence:
                continue
            detections.append(
                dict(
                    label=label,
                    confidence=certainty,
                    box=(x / width, y / height, box_width / width, box_height / height),
                    pixels=int(area),
                )
            )
    return detections


@dataclass
class PerceptionResult:
    mask: np.ndarray
    detections: list
    inference_ms: float
    total_ms: float
    captured_at: float


class SemanticPerception:
    def __init__(self, checkpoint, device="cuda", mixed_precision=False, input_size=None):
        import torch

        checkpoint = Path(checkpoint)
        if checkpoint.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Unexpectedly large checkpoint")
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = data["metadata"]
        validate_metadata(metadata)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision and device == "cuda"
        self.model = build_model(metadata["model"])
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.metadata = metadata
        self.size = input_size or (metadata["width"], metadata["height"])
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def infer(self, frame_bgr, captured_at=None):
        torch = self.torch
        start = time.perf_counter()
        captured_at = start if captured_at is None else captured_at
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize(
            self.size, Image.Resampling.BILINEAR
        )
        tensor = (
            torch.from_numpy(np.asarray(image).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
            / 255
        )
        tensor = (tensor - self.mean) / self.std
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, enabled=self.mixed_precision),
        ):
            logits = self.model(tensor)["out"]
            confidence, mask = logits.float().softmax(1).max(1)
            mask_array = mask[0].to("cpu").numpy().astype(np.uint8)
            confidence_array = confidence[0].to("cpu").numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        found = components(mask_array, confidence_array)
        return PerceptionResult(
            mask_array, found, inference_ms, (time.perf_counter() - start) * 1000, captured_at
        )


class LatestFrame:
    """One replaceable slot: inference cannot build a queue of stale frames."""

    def __init__(self):
        from threading import Condition

        self.condition = Condition()
        self.item = None
        self.closed = False
        self.dropped = 0

    def put(self, frame, timestamp):
        with self.condition:
            if self.closed:
                return
            if self.item is not None:
                self.dropped += 1
            self.item = (frame, timestamp)
            self.condition.notify()

    def get(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.item is not None or self.closed, timeout=timeout)
            item, self.item = self.item, None
            return item

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
