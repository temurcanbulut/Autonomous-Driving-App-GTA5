"""Development acceptance checks, not an automotive certification standard."""

import argparse
import json
from pathlib import Path


def gate(metrics):
    classes = {item["name"]: item for item in metrics["classes"]}
    checks = [
        ("mean_iou_at_least_0.65", metrics["mean_iou"] >= 0.65),
        ("person_pixel_recall_at_least_0.95", (classes["person"]["recall"] or 0) >= 0.95),
        ("car_iou_at_least_0.80", (classes["car"]["iou"] or 0) >= 0.80),
        ("bicycle_iou_at_least_0.50", (classes["bicycle"]["iou"] or 0) >= 0.50),
        (
            "bicycle_test_support_at_least_10000_pixels",
            classes["bicycle"]["support_pixels"] >= 10000,
        ),
        ("live_game_telemetry_and_input_verified", False),
        ("camera_and_ego_vehicle_mask_validated", False),
        ("end_to_end_perception_to_brake_evaluated", False),
    ]
    return dict(
        ready_for_live_assist=all(value for _, value in checks),
        checks=[dict(name=name, passed=passed) for name, passed in checks],
        scope="Project development criteria, deliberately separate from unit-test pass status. These thresholds are engineering targets, not validated safety probabilities.",
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--metrics", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = gate(json.loads(args.metrics.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
