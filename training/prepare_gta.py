"""Build a pinned, paired GTA dataset using the authors' official splits.

The upstream data is for research/education. Images are never added to Git.
"""

import argparse
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.io import loadmat

REPO = "roujin/GTA5subset"
REVISION = "05be1c51de63881d80425ba5f4ea702a83e7fddc"
SPLIT_URL = "https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip"
RAW_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
CLASSES = (
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


def get(url):
    response = requests.get(url, timeout=(10, 60))
    response.raise_for_status()
    return response


def entries(folder):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{folder}?limit=1000"
    result = []
    while url:
        response = get(url)
        result.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return {
        Path(item["path"]).stem: item
        for item in result
        if item["type"] == "file" and item["path"].endswith(".png")
    }


def fetch(item, root, source_root=None):
    if source_root is not None:
        cached = source_root / "source" / item["path"]
        expected = item.get("lfs", {}).get("oid")
        if (
            cached.is_file()
            and expected
            and hashlib.sha256(cached.read_bytes()).hexdigest() == expected
        ):
            return cached
    path = root / "source" / item["path"]
    expected = item.get("lfs", {}).get("oid")
    if path.is_file() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(4):
        try:
            response = get(
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}"
            )
            data = response.content
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Hash mismatch: {item['path']}")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(path)
            return path
        except (requests.RequestException, ValueError) as problem:
            error = problem
            time.sleep(0.5 * 2**attempt)
    raise RuntimeError(f"Failed download: {item['path']}") from error


def prepare_pair(identifier, image_entry, label_entry, root, splits, size, source_root=None):
    image_path, label_path = (
        fetch(image_entry, root, source_root),
        fetch(label_entry, root, source_root),
    )
    with Image.open(image_path) as image, Image.open(label_path) as label:
        if image.size != label.size:
            return {"id": identifier, "excluded": "image_mask_size_mismatch"}
        raw = np.asarray(label)
        if raw.ndim != 2:
            raise ValueError(f"Expected indexed labels: {identifier}, shape {raw.shape}")
        lut = np.full(256, 255, dtype=np.uint8)
        for index, raw_id in enumerate(RAW_IDS):
            lut[raw_id] = index
        encoded = lut[raw]
        image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        label = Image.fromarray(encoded).resize(size, Image.Resampling.NEAREST)
        relative_image = f"images/{identifier}.jpg"
        relative_label = f"labels/{identifier}.png"
        image.save(root / relative_image, quality=95)
        label.save(root / relative_label)
    partitions = [name for name, ids in splits.items() if int(identifier) in ids]
    if len(partitions) != 1:
        raise ValueError(f"Official split is not unique: {identifier}")
    return dict(
        id=identifier,
        split=partitions[0],
        image=relative_image,
        label=relative_label,
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=272)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for folder in ("images", "labels"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    archive = get(SPLIT_URL).content
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        data = loadmat(io.BytesIO(package.read("split.mat")))
    splits = {
        name: set(np.asarray(data[name + "Ids"]).reshape(-1).tolist())
        for name in ("train", "val", "test")
    }
    if any(
        splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Official splits overlap")
    images, labels = entries("images"), entries("labels")
    identifiers = sorted(images.keys() & labels.keys())
    print(json.dumps({"pairs": len(identifiers), "revision": REVISION}), flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                prepare_pair,
                identifier,
                images[identifier],
                labels[identifier],
                root,
                splits,
                (args.width, args.height),
                args.source_root,
            )
            for identifier in identifiers
        ]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 50 == 0:
                print(f"Prepared {index}/{len(futures)} pairs", flush=True)
    records.sort(key=lambda item: item["id"])
    manifest = dict(
        dataset=REPO,
        revision=REVISION,
        source="https://download.visinf.tu-darmstadt.de/data/from_games/",
        usage="research_and_education_only",
        split_source=SPLIT_URL,
        split_sha256=hashlib.sha256(archive).hexdigest(),
        width=args.width,
        height=args.height,
        classes=CLASSES,
        raw_ids=RAW_IDS,
        records=records,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({name: sum(item.get("split") == name for item in records) for name in splits}),
        flush=True,
    )


if __name__ == "__main__":
    main()
