"""Actual perception-to-dry-brake diagnostic on synthetic GTA cutout sequences.

This is NOT a gameplay or collision benchmark. Background motion, dynamics and
camera calibration are artificial. No game/OS input is imported or emitted.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Autonomous-Driving-App-GTA5"))
from autonomy.control import BrakeController, DryRunSink, Telemetry
from autonomy.geometry import CameraCalibration
from autonomy.hazards import estimate_hazards
from autonomy.perception import SemanticPerception
from autonomy.tracking import Tracker, iou


def sprite(data, records, class_id):
    # Choose detailed validation sprites for a visual diagnostic, not a random benchmark.
    for row in sorted(records, key=lambda x: (-x["class_pixels_original"][class_id], x["id"])):
        if row["class_pixels_original"][class_id] < 500:
            continue
        mask = np.asarray(Image.open(data / row["label"]))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), 8
        )
        if count < 2:
            continue
        selected = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        x, y, w, h, area = stats[selected]
        if area < 500 or h < 32 or w < 16:
            continue
        alpha = labels == selected
        if class_id == 18:
            # Keep an associated rider as visual context when present above bike.
            y0 = max(0, int(y - 2 * h))
            x0, x1 = max(0, int(x - w // 3)), min(mask.shape[1], int(x + w + w // 3))
            region = np.zeros(mask.shape, bool)
            region[y0 : y + h, x0:x1] = True
            alpha |= (mask == 12) & region
        ys, xs = np.nonzero(alpha)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        rgb = np.asarray(Image.open(data / row["image"]).convert("RGB"))
        rgba = np.dstack((rgb, alpha.astype(np.uint8) * 255))
        cutout = Image.fromarray(rgba).crop(box)
        # Rider context is visible, but bicycle matching uses bicycle pixels only.
        cutout.info["target_box"] = (
            float(x - box[0]) / cutout.width,
            float(y - box[1]) / cutout.height,
            float(w) / cutout.width,
            float(h) / cutout.height,
        )
        return cutout, row
    raise ValueError(f"No usable validation sprite for class {class_id}")


def render(background, actor, family, now):
    image = background.copy()
    if family == "no_inserted_actor":
        return image, None
    if family == "person_crossing":
        center, foot, height = -0.05 + 0.21 * now, 0.50 + 0.04 * now, 50 * (1 + 0.4 * now)
    elif family == "bicycle_crossing":
        center, foot, height = 0.99 - 0.17 * now, 0.50 + 0.04 * now, 60 * (1 + 0.4 * now)
    elif family == "vehicle_cut_in":
        if now < 0.6:
            return image, None
        center, foot, height = (
            max(0.5, 0.95 - 0.18 * (now - 0.6)),
            0.49 + 0.05 * now,
            48 * (1 + 0.4 * now),
        )
    else:
        center, foot, height = 0.06, 0.57, 90
    width = max(1, round(actor.width / actor.height * height))
    height = max(1, round(height))
    patch = actor.resize((width, height), Image.Resampling.BILINEAR)
    left, top = round(center * image.width - width / 2), round(foot * image.height - height)
    image.paste(patch, (left, top), patch)
    tx, ty, tw, th = actor.info.get("target_box", (0, 0, 1, 1))
    x0, y0, x1, y1 = (
        max(0, left + tx * width),
        max(0, top + ty * height),
        min(image.width, left + (tx + tw) * width),
        min(image.height, top + (ty + th) * height),
    )
    if x1 <= x0 or y1 <= y0:
        return image, None
    return image, (
        x0 / image.width,
        y0 / image.height,
        (x1 - x0) / image.width,
        (y1 - y0) / image.height,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--prepare-only", action="store_true")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists; preserve diagnostic records")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    validation = [
        r for r in manifest["records"] if r.get("split") == "val_challenge" and "excluded" not in r
    ]
    backgrounds = [
        r
        for r in manifest["records"]
        if r.get("split") == "val"
        and "excluded" not in r
        and sum(r["class_pixels_original"][11:]) == 0
        and r["class_pixels_original"][0] > 200000
    ]
    if not backgrounds:
        raise ValueError("No actor-free annotated validation background")
    background_row = sorted(backgrounds, key=lambda r: r["id"])[0]
    background = (
        Image.open(args.data / background_row["image"])
        .convert("RGB")
        .resize((960, 544), Image.Resampling.BILINEAR)
    )
    sprites = {
        name: sprite(args.data, validation, class_id)
        for name, class_id in [("person", 11), ("bicycle", 18), ("car", 13)]
    }
    source = dict(background=background_row, actors={k: v[1] for k, v in sprites.items()})
    (args.output / "sources.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    families = [
        ("person_crossing", "person"),
        ("bicycle_crossing", "bicycle"),
        ("vehicle_cut_in", "car"),
        ("sidewalk_person", "person"),
        ("no_inserted_actor", "person"),
    ]
    if args.prepare_only:
        for family, label in families:
            frame, _ = render(background, sprites[label][0], family, 2)
            frame.save(args.output / f"{family}.png")
        print(
            json.dumps(
                dict(status="prepared_cpu_only", sources=source["actors"].keys()), default=list
            ),
            flush=True,
        )
        return
    import torch

    torch.set_num_threads(4)
    model = SemanticPerception(args.model, "cuda")
    report = dict(
        protocol="synthetic_cutout_perception_to_dry_brake_v2_class_boxes",
        checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        data_split="validation_only",
        frames_per_sequence=40,
        fps=10,
        target_box_scope="Selected target-class pixels; rider context excluded from bicycle box.",
        game_input_emitted=False,
        limitations=[
            "Composited GTA sprites on a static GTA background, not actual gameplay.",
            "Media-clock timing and artificial flat-camera assumptions; no collision or stopping-efficacy result.",
            "No inserted actor is not a guarantee that unlabelled background pixels are actor-free.",
            "Confidence is not a calibrated safety probability.",
        ],
        sequences={},
    )
    for family, label in families:
        tracker, controller, sink = Tracker(), BrakeController(), DryRunSink()
        controller.arm()
        camera = CameraCalibration(1.5, 60, 0.4)
        reasons = Counter()
        frames_with_actor = detected = braking = 0
        first_detection = first_brake = None
        video = cv2.VideoWriter(
            str(args.output / f"{family}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 544)
        )
        if not video.isOpened():
            raise OSError("Video writer unavailable")
        try:
            with (args.output / f"{family}.jsonl").open("x", encoding="utf-8") as trace:
                for index in range(40):
                    now = index / 10
                    image, truth_box = render(background, sprites[label][0], family, now)
                    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                    result = model.infer(frame)
                    tracks = tracker.update(result.detections, now)
                    telemetry = Telemetry(12, now, True, False)
                    hazards, estimates = estimate_hazards(tracks, telemetry, camera, now)
                    decision = controller.tick(now, telemetry, hazards)
                    sink.apply(now, decision)
                    hit = truth_box is not None and any(
                        d["label"] == label and iou(d["box"], truth_box) >= 0.15
                        for d in result.detections
                    )
                    frames_with_actor += truth_box is not None
                    detected += hit
                    braking += sink.pressed
                    reasons[decision.reason] += 1
                    if hit and first_detection is None:
                        first_detection = now
                    if sink.pressed and first_brake is None:
                        first_brake = now
                    record = dict(
                        frame=index,
                        media_time_s=now,
                        truth_box=truth_box,
                        target_detected=hit,
                        detections=result.detections,
                        estimates=estimates,
                        decision=asdict(decision),
                        inference_ms=result.total_ms,
                        game_input_emitted=False,
                    )
                    trace.write(json.dumps(record) + "\n")
                    for detection in result.detections:
                        x, y, w, h = detection["box"]
                        cv2.rectangle(
                            frame,
                            (round(x * 960), round(y * 544)),
                            (round((x + w) * 960), round((y + h) * 544)),
                            (100, 220, 210),
                            1,
                        )
                    cv2.putText(
                        frame,
                        f"{family} | {now:.1f}s | DRY {decision.state}",
                        (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "SYNTHETIC DIAGNOSTIC | NO GAME INPUT | GTA data: Richter et al. 2016",
                        (15, 525),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )
                    video.write(frame)
                    if index in (0, 15, 30):
                        cv2.imwrite(str(args.output / f"{family}-{index:02d}.png"), frame)
        finally:
            video.release()
        report["sequences"][family] = dict(
            visible_frames=frames_with_actor,
            target_detected_frames=detected,
            first_detection_s=first_detection,
            first_any_dry_brake_s=first_brake,
            braking_frames=braking,
            reasons=dict(reasons),
        )
        print(json.dumps({family: report["sequences"][family]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
