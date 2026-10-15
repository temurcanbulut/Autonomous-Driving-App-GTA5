"""Evaluate runtime masks and semantic-component box proxies on frozen splits."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Autonomous-Driving-App-GTA5"))
sys.path.insert(0, str(ROOT / "training"))
from autonomy.perception import ACTORS, SemanticPerception, components
from autonomy.tracking import iou
from metrics import summarize

FOCUS = ("person", "rider", "car", "motorcycle", "bicycle")


def match_counts(predictions, truth, threshold):
    remaining = set(range(len(truth)))
    true_positive = 0
    for prediction in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
        overlap, index = max(
            ((iou(prediction["box"], truth[i]["box"]), i) for i in remaining), default=(0, None)
        )
        if overlap >= threshold:
            true_positive += 1
            remaining.remove(index)
    return dict(tp=true_positive, fp=len(predictions) - true_positive, fn=len(remaining))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--splits", nargs="+", default=["test", "test_challenge"])
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists")
    import torch

    torch.set_num_threads(4)
    perception = SemanticPerception(args.model, args.device)
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    result = dict(
        checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        model=perception.metadata["model"],
        input_size=perception.size,
        confidence_threshold=0.65,
        minimum_component_area_pixels=16,
        partitions={},
        limitation="Connected semantic regions are proxies, not true object instances. Counts are not COCO AP or driving accuracy; ignored ground truth regions cannot establish false positives.",
    )
    for split in args.splits:
        records = [
            r for r in manifest["records"] if r.get("split") == split and "excluded" not in r
        ]
        matrix = np.zeros((19, 19), np.int64)
        counts = {
            str(t): {name: dict(tp=0, fp=0, fn=0, unknown_predictions=0) for name in FOCUS}
            for t in (0.25, 0.5)
        }
        for index, row in enumerate(records, 1):
            frame = cv2.imread(str(args.data / row["image"]))
            prediction = perception.infer(frame)
            with Image.open(args.data / row["label"]) as image:
                truth = np.asarray(image.resize(perception.size, Image.Resampling.NEAREST))
            keep = truth != 255
            matrix += np.bincount(
                19 * truth[keep].astype(int) + prediction.mask[keep], minlength=361
            ).reshape(19, 19)
            true_boxes = components(truth, np.ones(truth.shape, np.float32))
            known_predictions, unknown = [], {name: 0 for name in FOCUS}
            for detection in prediction.detections:
                label = detection["label"]
                if label not in FOCUS:
                    continue
                x, y, w, h = detection["box"]
                height, width = truth.shape
                x0, y0, x1, y1 = (
                    round(x * width),
                    round(y * height),
                    round((x + w) * width),
                    round((y + h) * height),
                )
                class_id = next(k for k, v in ACTORS.items() if v == label)
                region = prediction.mask[y0:y1, x0:x1] == class_id
                known = (truth[y0:y1, x0:x1] != 255) & region
                if known.sum() < 0.5 * region.sum():
                    unknown[label] += 1
                else:
                    known_predictions.append(detection)
            for threshold in (0.25, 0.5):
                for name in FOCUS:
                    matched = match_counts(
                        [p for p in known_predictions if p["label"] == name],
                        [p for p in true_boxes if p["label"] == name],
                        threshold,
                    )
                    for key, value in matched.items():
                        counts[str(threshold)][name][key] += value
                    counts[str(threshold)][name]["unknown_predictions"] += unknown[name]
            if index % 100 == 0:
                print(
                    json.dumps(dict(split=split, evaluated=index, total=len(records))), flush=True
                )
        metrics = summarize(matrix, manifest["classes"])
        for values in counts.values():
            for value in values.values():
                value["precision"] = (
                    value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else None
                )
                value["recall"] = (
                    value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else None
                )
        result["partitions"][split] = dict(
            samples=len(records), pixel_metrics=metrics, component_proxy=counts
        )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                s: dict(samples=r["samples"], mean_iou=r["pixel_metrics"]["mean_iou"])
                for s, r in result["partitions"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
