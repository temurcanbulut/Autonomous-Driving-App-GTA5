"""Seeded offline longitudinal scenarios. This is not GTA or an end-to-end test."""

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
from autonomy.control import BrakeController, DryRunSink, Hazard, Telemetry


@dataclass(frozen=True)
class Scenario:
    seed: int
    family: str
    speed: float
    distance: float
    lead_speed: float
    brake_deceleration: float
    sensor_delay: float
    appearance_time: float
    lateral_speed: float


def generate(seed):
    rng = random.Random(seed)
    family = rng.choice(
        (
            "pedestrian_crossing",
            "vehicle_cut_in",
            "stopped_vehicle",
            "sidewalk_person",
            "clear_road",
            "late_unavoidable",
            "stale_sensor",
            "near_stop",
        )
    )
    speed = rng.uniform(7, 23) if family != "near_stop" else rng.uniform(0, 1.5)
    return Scenario(
        seed,
        family,
        speed,
        rng.uniform(12, 65),
        rng.uniform(0, speed * 0.65) if family == "vehicle_cut_in" else 0,
        rng.uniform(6, 10),
        rng.uniform(0, 0.10),
        rng.uniform(0, 0.5),
        rng.uniform(0.8, 2.5),
    )


def simulate(scenario, assist=True):
    dt = 0.02
    controller = BrakeController()
    controller.arm()
    sink = DryRunSink()
    speed, position = scenario.speed, 0.0
    minimum_speed = speed
    brake_seconds = 0.0
    history = deque([(0.0, speed, position)], maxlen=40)
    collided = False
    ticks = []
    for step in range(400):
        now = step * dt
        history.append((now, speed, position))
        observed = min(history, key=lambda item: abs(item[0] - (now - scenario.sensor_delay)))
        stamp, sensed_speed, sensed_position = observed
        obstacle_distance = scenario.distance + scenario.lead_speed * now - position
        sensed_distance = scenario.distance + scenario.lead_speed * stamp - sensed_position
        closing = max(0.01, sensed_speed - scenario.lead_speed)
        ttc = max(0.001, sensed_distance - 4.0) / closing
        visible = now >= scenario.appearance_time
        if scenario.family == "late_unavoidable":
            visible = obstacle_distance < max(2, speed * 0.25)
        if scenario.family == "vehicle_cut_in":
            lateral = max(0, 2.8 - scenario.lateral_speed * max(0, now - scenario.appearance_time))
            predicted_lateral = max(0, lateral - scenario.lateral_speed * max(0, ttc))
        elif scenario.family == "pedestrian_crossing":
            lateral = 2.8 - scenario.lateral_speed * max(0, now - scenario.appearance_time)
            predicted_lateral = lateral - scenario.lateral_speed * max(0, ttc)
        elif scenario.family in {"sidewalk_person", "clear_road"}:
            lateral = predicted_lateral = 4
        else:
            lateral = predicted_lateral = 0
        uncertainty = (
            scenario.lateral_speed * 0.3 if scenario.family == "pedestrian_crossing" else 0
        )
        hazard = visible and abs(predicted_lateral) < 1.2 + uncertainty and ttc > 0
        telemetry = Telemetry(
            sensed_speed,
            stamp - (0.4 if scenario.family == "stale_sensor" else 0),
            forward_gear=True,
            manual_override=False,
        )
        hazards = (
            [
                Hazard(
                    "person"
                    if "person" in scenario.family or "pedestrian" in scenario.family
                    else "vehicle",
                    ttc,
                    0.95,
                    stamp,
                    path_conflict=hazard,
                )
            ]
            if visible
            else []
        )
        decision = controller.tick(now, telemetry, hazards)
        if assist:
            sink.apply(now, decision)
        sink.advance(now)
        if sink.pressed:
            brake_seconds += dt
            speed -= scenario.brake_deceleration * dt
        elif speed > 0:
            speed = max(0, speed - 0.25 * dt)
        position += speed * dt
        minimum_speed = min(minimum_speed, speed)
        ticks.append(
            dict(
                time=round(now, 3),
                speed=round(speed, 4),
                distance=round(obstacle_distance, 4),
                lateral=round(lateral, 4),
                s_pressed=sink.pressed,
                state=decision.state,
            )
        )
        if obstacle_distance <= 1 and abs(lateral) < 1.2:
            collided = True
            break
    return dict(
        scenario=asdict(scenario),
        collision=collided,
        reverse=minimum_speed < -1e-6,
        min_speed=minimum_speed,
        brake_seconds=round(brake_seconds, 3),
        trace=ticks,
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--cases", type=int, default=2000)
    cli.add_argument("--seed", type=int, default=730000)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Preserve existing simulation results; choose a new output file.")
    groups = defaultdict(
        lambda: dict(
            cases=0, assist_collisions=0, baseline_collisions=0, reverse_events=0, brake_seconds=0.0
        )
    )
    failures = []
    for index in range(args.cases):
        scenario = generate(args.seed + index)
        assisted, baseline = simulate(scenario), simulate(scenario, assist=False)
        group = groups[scenario.family]
        group["cases"] += 1
        group["assist_collisions"] += assisted["collision"]
        group["baseline_collisions"] += baseline["collision"]
        group["reverse_events"] += assisted["reverse"]
        group["brake_seconds"] += assisted["brake_seconds"]
        if assisted["reverse"] or assisted["collision"] and not baseline["collision"]:
            failures.append(assisted)
    result = dict(
        protocol="offline_oracle_state_longitudinal_v1",
        seed=args.seed,
        cases=args.cases,
        timestep_s=0.02,
        horizon_s=8,
        groups=dict(groups),
        counterexamples=failures[:20],
        counterexample_count=len(failures),
        limitations=[
            "Uses exact simulated obstacle geometry, not camera predictions.",
            "Simplified flat-road dynamics; no GTA physics or keyboard actuation has been validated.",
            "Late hazards and stale sensors may remain unavoidable; report all cohorts.",
            "Collision counts are scenario results, not a real-world or game safety probability.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"counterexamples", "limitations"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
