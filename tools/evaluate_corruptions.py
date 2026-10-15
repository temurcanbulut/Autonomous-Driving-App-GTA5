"""Frozen-model robustness under explicit image corruptions; not GTA gameplay."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Autonomous-Driving-App-GTA5"))
sys.path.insert(0, str(ROOT / "training"))
from autonomy.perception import SemanticPerception
from metrics import summarize


def corrupt(frame, kind):
    if kind == "clean":
        return frame
    if kind == "brightness_0.5":
        return (frame.astype(np.float32) * 0.5).astype(np.uint8)
    if kind == "jpeg_quality_35":
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        if not ok:
            raise ValueError("JPEG encoder unavailable")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if kind == "horizontal_blur_9":
        kernel = np.zeros((9, 9), np.float32)
        kernel[4, :] = 1 / 9
        return cv2.filter2D(frame, -1, kernel)
    raise ValueError("Unknown corruption")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--split", default="test")
    cli.add_argument("--frames", type=int, default=200)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists")
    import torch

    torch.set_num_threads(4)
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [
        r for r in manifest["records"] if r.get("split") == args.split and "excluded" not in r
    ]
    if not records or args.frames < 1:
        cli.error("Nonempty split and positive frame count required")
    random.Random(20261016).shuffle(records)
    records = records[: args.frames]
    model = SemanticPerception(args.model, "cuda")
    results = {}
    for kind in ("clean", "brightness_0.5", "jpeg_quality_35", "horizontal_blur_9"):
        matrix = np.zeros((19, 19), np.int64)
        for row in records:
            frame = cv2.imread(str(args.data / row["image"]))
            predicted = model.infer(corrupt(frame, kind)).mask
            with Image.open(args.data / row["label"]) as image:
                truth = np.asarray(image.resize(model.size, Image.Resampling.NEAREST))
            keep = truth != 255
            matrix += np.bincount(
                19 * truth[keep].astype(int) + predicted[keep], minlength=361
            ).reshape(19, 19)
        results[kind] = summarize(matrix, manifest["classes"])
        print(
            json.dumps(dict(split=args.split, corruption=kind, mean_iou=results[kind]["mean_iou"])),
            flush=True,
        )
    report = dict(
        protocol="paired_synthetic_image_corruptions_v1",
        checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        image_ids=[r["id"] for r in records],
        split=args.split,
        input_size=model.size,
        seed=20261016,
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        results=results,
        limitations=[
            "Synthetic corruption levels on fixed labels, not measured weather or gameplay performance.",
            "JPEG recompression is not an evaluation of an actual YouTube video codec.",
            "Test corruptions do not select checkpoints or tune thresholds.",
        ],
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
