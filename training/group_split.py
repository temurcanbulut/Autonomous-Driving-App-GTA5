"""Explicit experimental split for a mirror containing only official val images.

This never pretends to be the original benchmark's train/test partition.
"""

import argparse
import hashlib
import json
from pathlib import Path


def partition(identifier, seed=20261015, block_size=100):
    group = int(identifier) // block_size
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    return ("test" if value < 0.20 else "val" if value < 0.35 else "train"), group


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output manifest exists; preserve experiment provenance.")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for item in data["records"]:
        if "excluded" in item:
            continue
        item["source_split"] = item["split"]
        item["split"], item["frame_group"] = partition(item["id"])
    data["split_protocol"] = (
        "Experimental SHA256-seeded assignment of 100-frame-ID blocks: 65% train, 15% val, 20% test. This mirror contains only original validation frames. Frame-ID proximity is a proxy, not verified gameplay-session separation. Not the official GTA benchmark."
    )
    data["split_seed"] = 20261015
    data["source_manifest_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        {
            split: sum(item.get("split") == split for item in data["records"])
            for split in ("train", "val", "test")
        }
    )


if __name__ == "__main__":
    main()
