"""Offline observations -> tracks -> approximate hazards -> dry-run brake.

No OS/game input is imported or emitted. Camera and telemetry must share an
explicit media clock; neither calibration nor dynamics is inferred from video.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
from autonomy.control import BrakeController, DryRunSink, Telemetry, finite_number
from autonomy.geometry import CameraCalibration
from autonomy.hazards import estimate_hazards
from autonomy.tracking import Tracker


def read_records(path):
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Replay input exceeds 128 MiB")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > 100000 or len(line) > 2 * 1024 * 1024:
                raise ValueError("Replay record limit exceeded")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Replay rows must be JSON objects")
            yield item


def checked_telemetry(rows):
    previous = -math.inf
    for row in rows:
        if (
            not isinstance(row, dict)
            or not finite_number(row.get("timestamp"))
            or not finite_number(row.get("speed_mps"))
        ):
            raise ValueError("Finite numeric telemetry is required")
        if row["timestamp"] <= previous:
            raise ValueError("Telemetry clock must be strictly increasing")
        for flag in ("forward_gear", "manual_override"):
            if not isinstance(row.get(flag), bool):
                raise ValueError("Gear and driver status must be explicit booleans")
        previous = row["timestamp"]
        yield row


def replay(observations, telemetry_rows, camera, arm=False):
    rows = iter(checked_telemetry(telemetry_rows))
    upcoming = next(rows, None)
    latest = None
    previous_sensor_time = previous_time = -math.inf
    tracker, controller, sink = Tracker(), BrakeController(), DryRunSink()
    if arm:
        controller.arm()
    for observation in observations:
        now = observation["media_time_s"]
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now <= previous_time:
            raise ValueError("Observation clock must be finite and strictly increasing")
        previous_time = now
        while upcoming is not None and upcoming["timestamp"] <= now:
            sensor_time = upcoming["timestamp"]
            if (
                not isinstance(sensor_time, (int, float))
                or not math.isfinite(sensor_time)
                or sensor_time <= previous_sensor_time
            ):
                raise ValueError("Telemetry clock must be finite and strictly increasing")
            previous_sensor_time = sensor_time
            for flag in ("forward_gear", "manual_override"):
                if flag in upcoming and not isinstance(upcoming[flag], bool):
                    raise ValueError("Telemetry flags must be booleans")
            latest = Telemetry(
                upcoming["speed_mps"],
                sensor_time,
                upcoming["forward_gear"],
                upcoming["manual_override"],
            )
            upcoming = next(rows, None)
        detections = observation.get("detections", [])
        if not isinstance(detections, list) or len(detections) > 2000:
            raise ValueError("Invalid detection list")
        tracks = tracker.update(detections, now)
        hazards, estimates = estimate_hazards(tracks, latest, camera, now)
        decision = controller.tick(now, latest, hazards)
        sink.apply(now, decision)
        yield dict(
            media_time_s=now,
            estimates=estimates,
            decision=asdict(decision),
            simulated_s_pressed=sink.pressed,
            game_input_emitted=False,
        )
    sink.advance(previous_time + 1)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--observations", type=Path, required=True)
    cli.add_argument("--telemetry", type=Path, required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--arm-dry-run", action="store_true")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output exists; preserve audit history")
    try:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        camera = CameraCalibration(
            **{k: calibration[k] for k in ("height_m", "vertical_fov_degrees", "horizon_y")}
        )
        camera.distance(0.8)
        header = dict(
            type="metadata",
            schema="gta-dry-control/1",
            game_input_emitted=False,
            observations_sha256=hashlib.sha256(args.observations.read_bytes()).hexdigest(),
            telemetry_sha256=hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
            calibration=calibration,
            assumptions="Fixed calibrated camera, flat ground, no occlusion compensation, synthetic S lease only.",
        )
        count = braking = 0
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(header) + "\n")
            for row in replay(
                read_records(args.observations),
                read_records(args.telemetry),
                camera,
                args.arm_dry_run,
            ):
                destination.write(json.dumps(row) + "\n")
                count += 1
                braking += row["simulated_s_pressed"]
            destination.write(
                json.dumps(
                    dict(
                        type="complete",
                        frames=count,
                        braking_frames=braking,
                        game_input_emitted=False,
                    )
                )
                + "\n"
            )
        print(json.dumps(dict(frames=count, braking_frames=braking, game_input_emitted=False)))
    except (ValueError, KeyError, TypeError, OSError) as error:
        cli.exit(2, f"Invalid replay: {error}\n")


if __name__ == "__main__":
    main()
