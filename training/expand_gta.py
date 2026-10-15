"""Acquire an auditable, rare-class enriched subset from the original GTA data.

Range requests avoid downloading 57 GiB of full image archives. ZIP members are
bounded and CRC-checked; images stay outside Git. No server-side code is run.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import struct
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from prepare_gta import CLASSES, RAW_IDS, SPLIT_URL
from scipy.io import loadmat

BASE = "https://download.visinf.tu-darmstadt.de/data/from_games/data/"


def request(url, headers=None):
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=(10, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RangeReader:
    """Minimal seekable view used only to read ZIP central directories."""

    def __init__(self, url):
        self.url = url
        first = request(url, {"Range": "bytes=0-63"})
        if first.status_code != 206:
            raise ValueError("Server does not support bounded HTTP range requests")
        self.size = int(first.headers["Content-Range"].split("/")[-1])
        self.etag = first.headers.get("ETag")
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 16 * 1024 * 1024:
            raise ValueError("Unexpectedly large ZIP metadata read")
        response = request(
            self.url, {"Range": f"bytes={self.position}-{self.position + count - 1}"}
        )
        if response.status_code != 206 or len(response.content) != count:
            raise ValueError("Invalid range response")
        if self.etag and response.headers.get("ETag") != self.etag:
            raise ValueError("Upstream archive changed during acquisition")
        self.position += count
        return response.content


def image_index(part):
    stream = RangeReader(f"{BASE}{part:02d}_images.zip")
    with zipfile.ZipFile(stream) as archive:
        items = {
            Path(x.filename).stem: x for x in archive.infolist() if x.filename.endswith(".png")
        }
    return stream, items


def fetch_member(stream, info):
    if info.file_size > 16 * 1024 * 1024 or info.compress_size > 16 * 1024 * 1024:
        raise ValueError("Unexpected image size")
    start = info.header_offset
    end = min(stream.size - 1, start + info.compress_size + 4095)
    response = request(stream.url, {"Range": f"bytes={start}-{end}"})
    if response.status_code != 206 or (stream.etag and response.headers.get("ETag") != stream.etag):
        raise ValueError("Archive identity/range mismatch")
    body = response.content
    if body[:4] != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP header")
    name_length, extra_length = struct.unpack_from("<HH", body, 26)
    offset = 30 + name_length + extra_length
    compressed = body[offset : offset + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError("Truncated member")
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, info.file_size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Invalid/bounded deflate member")
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError("Image CRC/length mismatch")
    return data


def acquire_labels(part, root):
    destination = root / "archives" / f"{part:02d}_labels.zip"
    if not destination.exists():
        data = request(f"{BASE}{part:02d}_labels.zip").content
        temporary = destination.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    rows = []
    with zipfile.ZipFile(destination) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".png"):
                continue
            data = archive.read(info)
            with Image.open(io.BytesIO(data)) as label:
                if label.mode not in ("P", "L"):
                    raise ValueError("Labels must be indexed")
                histogram = label.histogram()[:256]
                size = label.size
            rows.append(
                dict(
                    id=Path(info.filename).stem,
                    part=part,
                    label_member=info.filename,
                    source_size=size,
                    counts=[histogram[x] for x in RAW_IDS],
                    label_sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    return dict(
        part=part, archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), rows=rows
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--previous-manifest", type=Path, required=True)
    cli.add_argument("--train-size", type=int, default=6000)
    cli.add_argument("--eval-size", type=int, default=800)
    cli.add_argument("--workers", type=int, default=10)
    cli.add_argument("--labels-only", action="store_true")
    args = cli.parse_args()
    root = args.root.resolve()
    for folder in ("archives", "images", "labels", "source-images"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    catalogue_path = root / "label-catalogue.json"
    if catalogue_path.exists():
        catalogue = json.loads(catalogue_path.read_text())
    else:
        catalogue = []
        with concurrent.futures.ThreadPoolExecutor(5) as pool:
            for item in pool.map(lambda part: acquire_labels(part, root), range(1, 11)):
                catalogue.append(item)
                print(
                    json.dumps(dict(labels_part=item["part"], frames=len(item["rows"]))), flush=True
                )
        catalogue_path.write_text(json.dumps(catalogue), encoding="utf-8")
    rows = [r for part in catalogue for r in part["rows"]]
    print(
        json.dumps(
            dict(
                total_labels=len(rows),
                presence={
                    name: sum(r["counts"][i] >= 100 for r in rows) for i, name in enumerate(CLASSES)
                },
            )
        ),
        flush=True,
    )
    if args.labels_only:
        return
    split_bytes = request(SPLIT_URL).content
    with zipfile.ZipFile(io.BytesIO(split_bytes)) as package:
        raw_splits = loadmat(io.BytesIO(package.read("split.mat")))
    splits = {
        s: set(np.asarray(raw_splits[s + "Ids"]).reshape(-1).tolist())
        for s in ("train", "val", "test")
    }
    previous = json.loads(args.previous_manifest.read_text())
    previous_ids = {int(r["id"]) for r in previous["records"]}
    rng = random.Random(20261016)
    selected = []
    for split in ("train", "val", "test"):
        candidates = [
            r.copy()
            for r in rows
            if int(r["id"]) in splits[split] and int(r["id"]) not in previous_ids
        ]
        rng.shuffle(candidates)
        size = args.train_size if split == "train" else args.eval_size
        if split == "train":
            # All available rare VRUs, then pedestrians, then an ordinary-scene sample.
            rare = [r for r in candidates if any(r["counts"][i] >= 100 for i in (12, 17, 18))]
            people = [r for r in candidates if r["counts"][11] >= 500 and r not in rare]
            chosen = rare + people[: max(0, min(len(people), size * 2 // 3 - len(rare)))]
            remaining = [r for r in candidates if r not in chosen]
            chosen += remaining[: max(0, size - len(chosen))]
        else:
            # Random representative evaluation plus a disjoint targeted challenge pool.
            chosen = candidates[:size]
            pool = candidates[size:]
            bikes = [r for r in pool if r["counts"][18] >= 100]
            riders = [r for r in pool if r["counts"][12] >= 100 and r not in bikes][:100]
            pedestrians = [
                r for r in pool if r["counts"][11] >= 500 and r not in bikes and r not in riders
            ][:100]
            challenge = bikes + riders + pedestrians
            for row in challenge:
                row["split"] = split + "_challenge"
            selected.extend(challenge)
        for row in chosen:
            row["split"] = split
        selected.extend(chosen)
    plan = dict(
        seed=20261016,
        previous_excluded_ids=sorted(previous_ids),
        protocol="Official partitions; all 1000 previous experiment IDs excluded from every partition. Training rare-class enriched; val/test random samples plus separately reported rare-class challenges. Scene session separation is not established.",
        records=sorted(selected, key=lambda r: r["id"]),
    )
    plan_path = root / "acquisition-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Acquisition plan changed; use a new root")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                selected={
                    s: sum(r["split"] == s for r in selected)
                    for s in ("train", "val", "test", "val_challenge", "test_challenge")
                }
            )
        ),
        flush=True,
    )
    indexes = {}
    for part in sorted({r["part"] for r in selected}):
        indexes[part] = image_index(part)
    lut = np.full(256, 255, dtype=np.uint8)
    for i, raw in enumerate(RAW_IDS):
        lut[raw] = i

    def prepare(row):
        identifier = row["id"]
        metadata_path = root / "images" / f"{identifier}.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        stream, index = indexes[row["part"]]
        image_bytes = fetch_member(stream, index[identifier])
        with Image.open(io.BytesIO(image_bytes)) as image:
            if tuple(row["source_size"]) != image.size:
                return dict(id=identifier, excluded="source_image_label_size_mismatch")
            # Retain detail for random target-centered crops during training.
            image = image.convert("RGB").resize((1280, 720), Image.Resampling.BILINEAR)
            image.save(root / "images" / f"{identifier}.jpg", quality=95)
        with zipfile.ZipFile(root / "archives" / f"{row['part']:02d}_labels.zip") as archive:
            label_bytes = archive.read(row["label_member"])
        with Image.open(io.BytesIO(label_bytes)) as image:
            encoded = Image.fromarray(lut[np.asarray(image)]).resize(
                (1280, 720), Image.Resampling.NEAREST
            )
            encoded.save(root / "labels" / f"{identifier}.png")
        record = dict(
            id=identifier,
            split=row["split"],
            image=f"images/{identifier}.jpg",
            label=f"labels/{identifier}.png",
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            label_sha256=row["label_sha256"],
            prepared_image_sha256=hashlib.sha256(
                (root / "images" / f"{identifier}.jpg").read_bytes()
            ).hexdigest(),
            prepared_label_sha256=hashlib.sha256(
                (root / "labels" / f"{identifier}.png").read_bytes()
            ).hexdigest(),
            class_pixels_original=row["counts"],
            source_part=row["part"],
            source_image_archive_etag=stream.etag,
        )
        metadata_path.write_text(json.dumps(record), encoding="utf-8")
        return record

    records = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(prepare, row) for row in selected]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            records.append(future.result())
            if i % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            prepared=i,
                            total=len(selected),
                            seconds=round(time.perf_counter() - started, 1),
                        )
                    ),
                    flush=True,
                )
    manifest = dict(
        dataset="Playing for Data / original GTA5",
        revision="2016-original-archives-range-crc-sha256",
        source=BASE,
        usage="research_and_education_only",
        width=1280,
        height=720,
        classes=CLASSES,
        split_protocol=plan["protocol"],
        split_seed=plan["seed"],
        split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        label_archive_hashes={str(x["part"]): x["archive_sha256"] for x in catalogue},
        records=sorted(records, key=lambda r: r["id"]),
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(done=True, records=len(records))), flush=True)


if __name__ == "__main__":
    main()
