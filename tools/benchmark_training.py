"""Compare equivalent loss kernels/layouts on the same real training batch."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "Autonomous-Driving-App-GTA5"))
from autonomy.models import build_model
from train_focused import FocusFrames, focused_loss


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--reference-script", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--batch-size", type=int, default=4)
    cli.add_argument(
        "--modes",
        nargs="+",
        default=["legacy_contiguous", "vectorized_contiguous", "vectorized_channels_last"],
    )
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    spec = importlib.util.spec_from_file_location("frozen_training", args.reference_script)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(42)
    logits = torch.randn(2, 19, 8, 12, requires_grad=True)
    labels = torch.randint(0, 19, (2, 8, 12))
    labels[:, 0] = 255
    weights = torch.linspace(0.25, 5, 19)
    old_loss = reference.focused_loss(logits, labels, weights)
    old_gradient = torch.autograd.grad(old_loss, logits)[0]
    copied = logits.detach().clone().requires_grad_()
    new_loss = focused_loss(copied, labels, weights)
    new_gradient = torch.autograd.grad(new_loss, copied)[0]
    parity = dict(
        loss_abs_error=abs(float(old_loss.detach() - new_loss.detach())),
        gradient_max_abs_error=float((old_gradient - new_gradient).abs().max()),
    )
    if max(parity.values()) > 1e-6:
        raise ValueError("Loss/gradient parity failed")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    manifest = json.loads((args.data / "manifest.json").read_text())
    frames = FocusFrames(args.data, manifest["records"], "train", (meta["width"], meta["height"]))
    images, labels = next(iter(DataLoader(frames, batch_size=args.batch_size, num_workers=0)))
    labels, weights = labels.to("cuda"), torch.ones(19, device="cuda")
    results = []
    for name, loss_fn, layout in [
        ("legacy_contiguous", reference.focused_loss, torch.contiguous_format),
        ("vectorized_contiguous", focused_loss, torch.contiguous_format),
        ("vectorized_channels_last", focused_loss, torch.channels_last),
    ]:
        if name not in args.modes:
            continue
        model = build_model(meta["model"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device="cuda", memory_format=layout).train()
        inputs = images.to("cuda", memory_format=layout)
        scaler = torch.amp.GradScaler("cuda")
        times = []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(25):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda"):
                loss = loss_fn(model(inputs)["out"], labels, weights)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            if iteration >= 5:
                times.append((time.perf_counter() - started) * 1000)
        row = dict(
            mode=name,
            median_ms=statistics.median(times),
            p95_ms=sorted(times)[18],
            batch_size=args.batch_size,
            measured_steps=20,
            warmup_steps=5,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, inputs, loss, scaler
        torch.cuda.empty_cache()
    report = dict(
        parity=parity,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        input_size=[meta["width"], meta["height"]],
        results=results,
        scope="Same predecoded training batch; AMP forward + loss + backward; no optimizer update, loader, augmentation, validation or gameplay included.",
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
