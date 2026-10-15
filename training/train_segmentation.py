"""Train actual GTA semantic perception; select on validation, test once.

Run after prepare_gta.py. All outputs stay local until the audit is complete.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from metrics import summarize
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Frames(Dataset):
    def __init__(self, root, manifest, split, augment=False):
        self.root = root
        self.records = [item for item in manifest["records"] if item.get("split") == split]
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as loaded:
            image = loaded.convert("RGB")
        with Image.open(self.root / record["label"]) as loaded:
            label = loaded.copy()
        if self.augment:
            if random.random() < 0.5:
                image, label = ImageOps.mirror(image), ImageOps.mirror(label)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return (image - MEAN) / STD, torch.from_numpy(np.asarray(label).copy()).long()


def model_for(classes, pretrained):
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
    previous_low = model.classifier.low_classifier
    previous_high = model.classifier.high_classifier
    model.classifier.low_classifier = nn.Conv2d(previous_low.in_channels, len(classes), 1)
    model.classifier.high_classifier = nn.Conv2d(previous_high.in_channels, len(classes), 1)
    if weights:
        old_names = weights.meta["categories"]
        aliases = {"motorcycle": "motorbike"}
        with torch.no_grad():
            for index, name in enumerate(classes):
                name = aliases.get(name, name)
                if name in old_names:
                    previous = old_names.index(name)
                    for new, old in (
                        (model.classifier.low_classifier, previous_low),
                        (model.classifier.high_classifier, previous_high),
                    ):
                        new.weight[index].copy_(old.weight[previous])
                        new.bias[index].copy_(old.bias[previous])
    return model


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64, device=device)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(images)["out"].argmax(1)
        keep = labels != 255
        matrix += torch.bincount(
            len(classes) * labels[keep] + prediction[keep], minlength=len(classes) ** 2
        ).reshape(len(classes), len(classes))
    return summarize(matrix.cpu().numpy(), classes)


def class_weights(dataset, count):
    histogram = np.zeros(count, dtype=np.int64)
    for record in dataset.records:
        with Image.open(dataset.root / record["label"]) as image:
            pixels = np.asarray(image)
            histogram += np.bincount(pixels[pixels < count], minlength=count)
    frequencies = histogram / max(1, histogram.sum())
    weights = 1 / np.sqrt(np.maximum(frequencies, 0.001))
    weights = np.clip(weights / weights.mean(), 0.25, 4)
    return torch.tensor(weights, dtype=torch.float32), histogram.tolist()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--init-checkpoint", type=Path)
    cli.add_argument("--balanced-sampling", action="store_true")
    cli.add_argument("--epochs", type=int, default=30)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--workers", type=int, default=2)
    cli.add_argument("--seed", type=int, default=20261015)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("A working CUDA GPU is required for this training run.")
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Use an empty output directory to preserve existing experiment evidence.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    manifest_bytes = (args.manifest or args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    classes = manifest["classes"]
    datasets = {
        split: Frames(args.data, manifest, split, split == "train")
        for split in ("train", "val", "test")
    }
    if any(not len(dataset) for dataset in datasets.values()):
        cli.error("Every official partition must contain samples.")
    sampler = None
    if args.balanced_sampling:
        presence = []
        for record in datasets["train"].records:
            with Image.open(args.data / record["label"]) as image:
                counts = np.bincount(np.asarray(image).reshape(-1), minlength=256)[: len(classes)]
                presence.append(counts >= 20)
        presence = np.asarray(presence)
        inverse = 1 / np.maximum(presence.sum(0), 1)
        sample_weights = (presence * inverse).sum(1)
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    metadata = dict(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        width=manifest["width"],
        height=manifest["height"],
        model="lraspp_mobilenet_v3_large",
        classes=classes,
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        dataset=manifest["dataset"],
        dataset_revision=manifest["revision"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        partitions={
            split: [item["id"] for item in dataset.records] for split, dataset in datasets.items()
        },
        limitations=[
            "Subset of a research/education GTA dataset, not the complete benchmark.",
            "Pixel accuracy is dominated by common classes; inspect per-class IoU and support.",
            "Official image splits do not establish separation by unseen gameplay session.",
            "Segmentation does not learn control actions, vehicle dynamics or calibrated depth.",
            "Test images are never used for gradient updates or checkpoint selection.",
        ],
    )
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["split_protocol"] = manifest.get("split_protocol", "Official split intersection")
    metadata["balanced_sampling"] = args.balanced_sampling
    metadata["training_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["limitations"][2] = (
        "Frame-ID groups do not establish separation by unseen gameplay session. See split_protocol; this run is not the official benchmark."
    )
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gpu": metadata["gpu"],
                "samples": {key: len(value) for key, value in datasets.items()},
            }
        ),
        flush=True,
    )
    model = model_for(classes, pretrained=True).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(initial["state_dict"])
        metadata["initial_checkpoint_sha256"] = hashlib.sha256(
            args.init_checkpoint.read_bytes()
        ).hexdigest()
    (args.output / "training-script.py").write_bytes(Path(__file__).read_bytes())
    weights, histogram = class_weights(datasets["train"], len(classes))
    metadata["train_class_pixels"] = histogram
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 0.0001}, {"params": head_params, "lr": 0.001}],
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    baseline = evaluate(model, loaders["val"], device, classes)
    (args.output / "initial-validation.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    best = -1.0
    started = time.perf_counter()
    log = (args.output / "training.jsonl").open("w", encoding="utf-8")
    try:
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = steps = 0
            for images, labels in loaders["train"]:
                images, labels = (
                    images.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    loss = criterion(model(images)["out"], labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                steps += 1
            scheduler.step()
            validation = evaluate(model, loaders["val"], device, classes)
            record = dict(
                epoch=epoch + 1,
                train_loss=total_loss / steps,
                validation_mean_iou=validation["mean_iou"],
                validation_pixel_accuracy=validation["pixel_accuracy"],
                seconds=round(time.perf_counter() - epoch_started, 2),
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if validation["mean_iou"] > best:
                best = validation["mean_iou"]
                metadata["best_epoch"] = epoch + 1
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "metadata": metadata,
                    },
                    args.output / "best.pt",
                )
                (args.output / "best-validation.json").write_text(
                    json.dumps(validation, indent=2), encoding="utf-8"
                )
    finally:
        log.close()
    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test = evaluate(model, loaders["test"], device, classes)
    test.update(
        dict(
            split="held_out_subset_groups"
            if "split_protocol" in manifest
            else "official_test_intersection",
            samples=len(datasets["test"]),
            selected_epoch=metadata["best_epoch"],
            training_seconds=round(time.perf_counter() - started, 2),
            checkpoint_sha256=hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest(),
        )
    )
    (args.output / "test-metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "test_mean_iou": test["mean_iou"],
                "test_pixel_accuracy": test["pixel_accuracy"],
                "selected_epoch": test["selected_epoch"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
