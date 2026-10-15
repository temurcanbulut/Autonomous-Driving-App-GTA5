"""Compare the same trained model/data on CPU and CUDA, with explicit scope."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
from autonomy.perception import SemanticPerception


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--frames", type=int, default=30)
    cli.add_argument("--split", default="test", help="use val for pre-test deployment comparisons")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Preserve prior benchmark evidence; use a new output file.")
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    records = [item for item in manifest["records"] if item.get("split") == args.split][
        : args.frames
    ]
    if not records or args.frames < 1:
        cli.error("A nonempty split and positive frame count are required")
    frames = [cv2.imread(str(args.data / record["image"])) for record in records]
    results = []
    for device, mixed in (("cpu", False), ("cuda", False), ("cuda", True)):
        perception = SemanticPerception(args.model, device=device, mixed_precision=mixed)
        for frame in frames[:5]:
            perception.infer(frame)
        measured = [perception.infer(frame) for frame in frames]
        latency = np.array([item.total_ms for item in measured])
        inference = np.array([item.inference_ms for item in measured])
        results.append(
            dict(
                device=device,
                autocast=mixed,
                frames=len(frames),
                total_p50_ms=float(np.median(latency)),
                total_p95_ms=float(np.percentile(latency, 95)),
                model_and_mask_transfer_p50_ms=float(np.median(inference)),
                throughput_from_mean_ms=float(1000 / latency.mean()),
            )
        )
        del perception
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = dict(
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        image_ids=[record["id"] for record in records],
        split=args.split,
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
        results=results,
        scope="Batch-one perception including RGB conversion, preprocessing, GPU synchronization/transfer, mask probabilities, connected components. Five warm-up frames excluded. Decoded images are in memory; excludes screen capture, video decoding, preview and game rendering. Not a game FPS claim.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
