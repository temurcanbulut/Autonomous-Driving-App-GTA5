"""Fine-tune perception on expanded labels, with target crops and focal/Dice loss.

Checkpoint selection uses representative and rare-class validation only. Test
metrics are written after the full training schedule. This learns semantic masks,
not brake actions or physical time-to-collision.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from train_segmentation import MEAN, STD, model_for

FOCUS = (11, 12, 13, 17, 18)


def crop_box(mask, rng, size):
    """Choose a spatial crop using training labels only, without changing labels."""
    height, width = mask.shape
    scale = rng.uniform(0.5, 1.0)
    crop_width = max(32, min(width, int(width * scale)))
    crop_height = max(32, min(height, round(crop_width * size[1] / size[0])))
    priority = [
        class_id for class_id in (18, 12, 17, 11) if np.count_nonzero(mask == class_id) >= 8
    ]
    if priority and rng.random() < 0.75:
        # Inverse-frequency sampler brings rare frames here; favor bicycles within them.
        class_id = priority[0] if rng.random() < 0.6 else rng.choice(priority)
        ys, xs = np.nonzero(mask == class_id)
        selected = rng.randrange(len(xs))
        center_x, center_y = int(xs[selected]), int(ys[selected])
        x = center_x - crop_width // 2 + rng.randint(-crop_width // 5, crop_width // 5)
        y = center_y - crop_height // 2 + rng.randint(-crop_height // 5, crop_height // 5)
    else:
        x, y = rng.randint(0, width - crop_width), rng.randint(0, height - crop_height)
    x = max(0, min(width - crop_width, x))
    y = max(0, min(height - crop_height, y))
    return x, y, x + crop_width, y + crop_height


class FocusFrames(Dataset):
    def __init__(self, root, records, split, size):
        self.root, self.size = root, size
        self.records = [r for r in records if r.get("split") == split and "excluded" not in r]
        self.training = split == "train"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["image"]) as loaded:
            image = loaded.convert("RGB")
        with Image.open(self.root / row["label"]) as loaded:
            label = loaded.copy()
        if self.training:
            if random.random() < 0.85:
                box = crop_box(np.asarray(label), random, self.size)
                image, label = image.crop(box), label.crop(box)
            if random.random() < 0.5:
                image, label = ImageOps.mirror(image), ImageOps.mirror(label)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.3))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
        image = image.resize(self.size, Image.Resampling.BILINEAR)
        label = label.resize(self.size, Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return (tensor - MEAN) / STD, torch.from_numpy(np.asarray(label).copy()).long()


def focused_loss(logits, target, weights):
    valid = target != 255
    # One log-softmax, no per-class host/GPU synchronization.
    safe_target = target.clamp(0, logits.shape[1] - 1)
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    selected = log_probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    focal = (
        (1 - selected.exp()) * -selected * weights[safe_target] * valid
    ).sum() / valid.sum().clamp_min(1)
    ids = target.new_tensor(FOCUS)
    truth = (target.unsqueeze(1) == ids.view(1, -1, 1, 1)).float()
    predicted = log_probabilities[:, FOCUS].exp() * valid.unsqueeze(1)
    axes = (0, 2, 3)
    support = truth.sum(axes)
    present = (support > 0).float()
    dice = 1 - (2 * (predicted * truth).sum(axes) + 1) / (predicted.sum(axes) + support + 1)
    return focal + 0.35 * (dice * present).sum() / present.sum().clamp_min(1)


def selection_score(metrics, focused_metrics=None):
    classes = {x["name"]: x for x in (focused_metrics or metrics)["classes"]}
    target_iou = np.mean(
        [(classes[x]["iou"] or 0) for x in ("person", "rider", "car", "motorcycle", "bicycle")]
    )
    return float(0.4 * metrics["mean_iou"] + 0.6 * target_iou)


def distillation_loss(student, teacher, target, temperature=2.0):
    valid = target != 255
    log_student = F.log_softmax(student.float() / temperature, dim=1)
    teacher_probabilities = F.softmax(teacher.detach().float() / temperature, dim=1)
    pixel = F.kl_div(log_student, teacher_probabilities, reduction="none").sum(1)
    return (pixel * valid).sum() / valid.sum().clamp_min(1) * temperature**2


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--init-checkpoint", type=Path, required=True)
    cli.add_argument(
        "--architecture",
        choices=("lraspp_mobilenet_v3_large", "deeplabv3_mobilenet_v3_large"),
        default="lraspp_mobilenet_v3_large",
    )
    cli.add_argument("--epochs", type=int, default=40)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--width", type=int, default=960)
    cli.add_argument("--height", type=int, default=544)
    cli.add_argument("--eval-width", type=int, default=0)
    cli.add_argument("--eval-height", type=int, default=0)
    cli.add_argument(
        "--defer-test",
        action="store_true",
        help="keep test predictions unopened during intermediate stages",
    )
    cli.add_argument("--backbone-lr", type=float, default=0.00015)
    cli.add_argument("--head-lr", type=float, default=0.0006)
    cli.add_argument("--teacher-checkpoint", type=Path)
    cli.add_argument("--seed", type=int, default=20261016)
    cli.add_argument(
        "--channels-last", action="store_true", help="use a measured channels-last CUDA layout"
    )
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Use a new output directory")
    if min(args.epochs, args.batch_size, args.width, args.height) <= 0:
        cli.error("Positive parameters required")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str((args.output.parent / "pretrained-cache").resolve()))
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    classes = manifest["classes"]
    partitions = ("train", "val", "test", "val_challenge", "test_challenge")
    evaluation_size = (args.eval_width or args.width, args.eval_height or args.height)
    datasets = {
        s: FocusFrames(
            args.data,
            manifest["records"],
            s,
            (args.width, args.height) if s == "train" else evaluation_size,
        )
        for s in partitions
    }
    if any(not len(x) for x in datasets.values()):
        cli.error("Every evaluation partition must have records")
    identities = [{r["id"] for r in d.records} for d in datasets.values()]
    if sum(map(len, identities)) != len(set.union(*identities)):
        cli.error("Partition IDs overlap")
    counts = np.array([r["class_pixels_original"] for r in datasets["train"].records])
    histogram = counts.sum(0)
    frequencies = histogram / histogram.sum()
    weights = 1 / np.sqrt(np.maximum(frequencies, 0.0001))
    weights = torch.tensor(
        np.clip(weights / weights.mean(), 0.25, 5), dtype=torch.float32, device="cuda"
    )
    sample_weights = (
        1
        + 1.5 * (counts[:, 11] >= 500)
        + 4 * (counts[:, 12] >= 100)
        + 3 * (counts[:, 17] >= 100)
        + 12 * (counts[:, 18] >= 100)
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            sampler=sampler if s == "train" else None,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0 and s in ("train", "val", "val_challenge"),
            drop_last=s == "train",
        )
        for s, d in datasets.items()
    }
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    if tuple(initial["metadata"]["classes"]) != tuple(classes):
        cli.error("Initial checkpoint class order differs")
    transfer = "all_parameters_from_initial_checkpoint"
    pretrained_source = None
    if args.architecture == "lraspp_mobilenet_v3_large":
        model = model_for(classes, pretrained=False)
        model.load_state_dict(initial["state_dict"])
    elif initial["metadata"]["model"] == args.architecture:
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        model = deeplabv3_mobilenet_v3_large(
            num_classes=len(classes), weights=None, weights_backbone=None, aux_loss=False
        )
        model.load_state_dict(initial["state_dict"])
    else:
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        pretrained_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(
            weights=pretrained_weights, weights_backbone=None, progress=False
        )
        old = model.classifier[-1]
        new = torch.nn.Conv2d(old.in_channels, len(classes), 1)
        with torch.no_grad():
            for i, name in enumerate(classes):
                alias = "motorbike" if name == "motorcycle" else name
                if alias in pretrained_weights.meta["categories"]:
                    source = pretrained_weights.meta["categories"].index(alias)
                    new.weight[i].copy_(old.weight[source])
                    new.bias[i].copy_(old.bias[source])
        model.classifier[-1] = new
        model.aux_classifier = None
        weight_file = (
            Path(torch.hub.get_dir()) / "checkpoints" / pretrained_weights.url.rsplit("/", 1)[-1]
        )
        pretrained_source = dict(
            enum=pretrained_weights.name,
            url=pretrained_weights.url,
            sha256=hashlib.sha256(weight_file.read_bytes()).hexdigest(),
        )
        if initial["metadata"]["model"] == args.architecture:
            model.load_state_dict(initial["state_dict"])
        else:
            backbone = {
                k.removeprefix("backbone."): v
                for k, v in initial["state_dict"].items()
                if k.startswith("backbone.")
            }
            model.backbone.load_state_dict(backbone, strict=True)
            transfer = "GTA LRASPP backbone; official COCO/VOC DeepLab head with shared-class channels transferred"
    model.to("cuda")
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model.to(memory_format=memory_format)
    teacher = None
    teacher_metadata = None
    if args.teacher_checkpoint:
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        if (
            tuple(checkpoint["metadata"]["classes"]) != tuple(classes)
            or checkpoint["metadata"]["model"] != "deeplabv3_mobilenet_v3_large"
        ):
            cli.error("Teacher must be a supported DeepLab checkpoint with matching classes")
        teacher = deeplabv3_mobilenet_v3_large(
            num_classes=len(classes), weights=None, weights_backbone=None, aux_loss=False
        )
        teacher.load_state_dict(checkpoint["state_dict"])
        teacher.to("cuda").eval().requires_grad_(False)
        teacher_metadata = dict(
            sha256=hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest(),
            model=checkpoint["metadata"]["model"],
            temperature=2.0,
            alpha=0.3,
            only_labelled_pixels=True,
        )
    metadata = dict(
        model=args.architecture,
        transfer=transfer,
        pretrained_source=pretrained_source,
        classes=classes,
        width=evaluation_size[0],
        height=evaluation_size[1],
        training_width=args.width,
        training_height=args.height,
        test_deferred=args.defer_test,
        learning_rates=dict(backbone=args.backbone_lr, head=args.head_lr),
        distillation=teacher_metadata,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_drop_last=True,
        training_memory_format="channels_last" if args.channels_last else "contiguous",
        dataset=manifest["dataset"],
        dataset_revision=manifest["revision"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        initial_checkpoint_sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        initial_torch_version=initial["metadata"].get("torch"),
        torch=str(torch.__version__),
        torchvision=__import__("torchvision").__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        partitions={s: [r["id"] for r in d.records] for s, d in datasets.items()},
        train_class_pixels=histogram.tolist(),
        selection="0.4 representative-validation mean IoU + 0.6 rare-validation mean IoU(person,rider,car,motorcycle,bicycle); test partitions never select checkpoints",
        split_protocol=manifest["split_protocol"],
        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loss="class-weighted focal gamma=1 plus 0.35 soft Dice over present focus classes",
        evaluation_precision="FP32, same normalization/resize as runtime",
        limitations=[
            "Semantic perception only, no learned control or calibrated distance.",
            "Frame-ID official partitions do not prove unseen-session separation.",
            "Rare-class challenge metrics have a different distribution and are reported separately.",
        ],
    )
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "training-script.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(dict(gpu=metadata["gpu"], samples={s: len(d) for s, d in datasets.items()})),
        flush=True,
    )
    # FP32 evaluation: the legacy helper's autocast is disabled explicitly here.
    device = torch.device("cuda")

    def assess(loader):
        from metrics import summarize

        model.eval()
        matrix = torch.zeros((19, 19), dtype=torch.int64, device=device)
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = (
                    images.to(device, non_blocking=True, memory_format=memory_format),
                    labels.to(device, non_blocking=True),
                )
                prediction = model(images)["out"].argmax(1)
                keep = labels != 255
                matrix += torch.bincount(
                    19 * labels[keep] + prediction[keep], minlength=361
                ).reshape(19, 19)
        return summarize(matrix.cpu().numpy(), classes)

    baseline = assess(loaders["val"])
    baseline_focused = assess(loaders["val_challenge"])
    (args.output / "initial-validation-challenge.json").write_text(
        json.dumps(baseline_focused, indent=2), encoding="utf-8"
    )
    (args.output / "initial-validation.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    best_score = selection_score(baseline, baseline_focused)
    metadata["best_epoch"] = 0
    torch.save(
        dict(
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            metadata=metadata,
        ),
        args.output / "best.pt",
    )
    (args.output / "best-validation.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    (args.output / "best-validation-challenge.json").write_text(
        json.dumps(baseline_focused, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    with (args.output / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            step_started = time.perf_counter()
            model.train()
            total_loss = steps = 0
            for images, labels in loaders["train"]:
                images, labels = (
                    images.to(device, non_blocking=True, memory_format=memory_format),
                    labels.to(device, non_blocking=True),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    logits = model(images)["out"]
                    loss = focused_loss(logits, labels, weights)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(images)["out"]
                        loss = 0.7 * loss + 0.3 * distillation_loss(logits, teacher_logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                steps += 1
                if steps % 100 == 0:
                    (args.output / "progress.json").write_text(
                        json.dumps(
                            dict(
                                epoch=epoch,
                                steps=steps,
                                total_steps=len(loaders["train"]),
                                mean_loss=total_loss / steps,
                                seconds=time.perf_counter() - step_started,
                            )
                        ),
                        encoding="utf-8",
                    )
            scheduler.step()
            validation = assess(loaders["val"])
            validation_focused = assess(loaders["val_challenge"])
            score = selection_score(validation, validation_focused)
            focus = {
                c["name"]: c["iou"]
                for c in validation["classes"]
                if c["name"] in ("person", "rider", "car", "motorcycle", "bicycle")
            }
            record = dict(
                epoch=epoch,
                train_loss=total_loss / steps,
                validation_mean_iou=validation["mean_iou"],
                validation_focus_iou=focus,
                validation_challenge_mean_iou=validation_focused["mean_iou"],
                validation_challenge_focus_iou={
                    c["name"]: c["iou"]
                    for c in validation_focused["classes"]
                    if c["name"] in ("person", "rider", "car", "motorcycle", "bicycle")
                },
                selection_score=score,
                seconds=round(time.perf_counter() - step_started, 2),
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score = score
                metadata["best_epoch"] = epoch
                torch.save(
                    dict(
                        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        metadata=metadata,
                    ),
                    args.output / "best.pt",
                )
                (args.output / "best-validation.json").write_text(
                    json.dumps(validation, indent=2), encoding="utf-8"
                )
                (args.output / "best-validation-challenge.json").write_text(
                    json.dumps(validation_focused, indent=2), encoding="utf-8"
                )
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    metadata["best_epoch"] = best["metadata"]["best_epoch"]
    metadata["training_seconds"] = round(time.perf_counter() - started, 2)
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for split in (
        ("val_challenge",) if args.defer_test else ("val_challenge", "test", "test_challenge")
    ):
        metrics = assess(loaders[split])
        metrics.update(
            split=split,
            samples=len(datasets[split]),
            selected_epoch=metadata["best_epoch"],
            checkpoint_sha256=hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest(),
        )
        (args.output / f"{split}-metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(dict(split=split, mean_iou=metrics["mean_iou"])), flush=True)
    (args.output / "status.json").write_text(
        json.dumps(
            dict(
                status="complete", completed_epochs=args.epochs, test_evaluated=not args.defer_test
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
