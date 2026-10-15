"""Verify downloaded assets and disjoint IDs/source hashes before GPU training."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def audit(data, manifest, previous):
    errors, splits, owners = [], defaultdict(list), {}
    old_ids = {r["id"] for r in previous["records"]}
    old_hashes = {r.get("image_sha256") for r in previous["records"]}
    for row in manifest["records"]:
        if "excluded" in row:
            continue
        split, identifier = row["split"], row["id"]
        if identifier in old_ids or row["image_sha256"] in old_hashes:
            errors.append(dict(id=identifier, reason="previous_experiment_overlap"))
        for field in ("image", "label"):
            path = (data / row[field]).resolve()
            if not path.is_relative_to(data.resolve()):
                errors.append(dict(id=identifier, reason="prepared_path_outside_data"))
                continue
            expected = row[f"prepared_{field}_sha256"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(dict(id=identifier, reason=f"{field}_integrity_mismatch"))
        for kind, digest in (
            ("id", identifier),
            ("source", row["image_sha256"]),
            ("prepared", row["prepared_image_sha256"]),
        ):
            key = kind, digest
            if key in owners and owners[key] != split:
                errors.append(
                    dict(
                        id=identifier,
                        reason="cross_partition_" + kind + "_duplicate",
                        other=owners[key],
                    )
                )
            owners[key] = split
        splits[split].append(row)
    classes = manifest["classes"]
    return dict(
        passed=not errors,
        errors=errors,
        partitions={
            split: dict(
                samples=len(rows),
                frame_presence={
                    name: sum(r["class_pixels_original"][i] >= 100 for r in rows)
                    for i, name in enumerate(classes)
                },
            )
            for split, rows in splits.items()
        },
        excluded=sum("excluded" in r for r in manifest["records"]),
        scope="Exact ID/byte hash separation; does not prove different gameplay-session separation.",
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--previous-manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists")
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    result = audit(
        args.data, json.loads(manifest_bytes), json.loads(args.previous_manifest.read_text())
    )
    result["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
