"""Deterministic brake/reverse interlock, exercised only through dry-run sinks."""

import math
from dataclasses import dataclass, replace
from numbers import Real

ACTOR_KINDS = {
    "person",
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "vehicle",
}


def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Telemetry:
    speed_mps: float
    timestamp: float
    forward_gear: bool | None = None
    manual_override: bool | None = None


@dataclass(frozen=True)
class Hazard:
    kind: str
    collision_time_s: float
    confidence: float
    timestamp: float
    path_conflict: bool = True


@dataclass(frozen=True)
class Decision:
    s_pressed: bool
    state: str
    reason: str
    throttle: bool = False
    release_by: float | None = None


class BrakeController:
    """Short digital brake pulses; release before S could select reverse.

    The stopping-speed bound assumes a known upper deceleration. It cannot
    replace calibrated game telemetry, latency measurements or a live test.
    """

    def __init__(self, max_age=0.15, max_deceleration=12.0):
        if not (
            math.isfinite(max_age)
            and 0 < max_age <= 1
            and math.isfinite(max_deceleration)
            and 0 < max_deceleration <= 30
        ):
            raise ValueError("Finite positive latency/deceleration bounds are required")
        self.max_age = max_age
        self.max_deceleration = max_deceleration
        self.armed = False
        self.last_time = None
        self.pulse_until = -math.inf
        self.release_until = -math.inf

    def arm(self):
        self.armed = True
        self.last_time = None
        self.pulse_until = self.release_until = -math.inf

    def disarm(self):
        self.armed = False
        self.pulse_until = self.release_until = -math.inf

    def fault(self, reason):
        self.disarm()
        return Decision(False, "FAULT", reason)

    def decide(self, now, telemetry, hazards):
        if not self.armed:
            return Decision(False, "DISARMED", "explicit_arm_required")
        if telemetry is None:
            return self.fault("speed_telemetry_missing")
        values = (now, telemetry.speed_mps, telemetry.timestamp)
        if not all(finite_number(value) for value in values):
            return self.fault("nonfinite_telemetry")
        if telemetry.manual_override is True:
            self.disarm()
            return Decision(False, "DISARMED", "manual_override")
        if telemetry.manual_override is not False:
            return self.fault("manual_override_status_unknown")
        if telemetry.forward_gear is not True or telemetry.speed_mps < 0:
            return self.fault("not_forward_motion")
        age = now - telemetry.timestamp
        if age < 0 or age > self.max_age:
            return self.fault("stale_or_future_telemetry")
        if self.last_time is not None and (
            now <= self.last_time or now - self.last_time > self.max_age
        ):
            self.last_time = now
            self.pulse_until = self.release_until = -math.inf
            return self.fault("control_clock_discontinuity")
        self.last_time = now
        # Protect against residual braking during sensor/actuation delay.
        stop_margin = 0.25 + self.max_deceleration * age
        if telemetry.speed_mps <= stop_margin:
            self.pulse_until = -math.inf
            return Decision(False, "STOPPED", "release_s_before_reverse")
        valid = [
            item
            for item in hazards
            if item.path_conflict is True
            and item.kind in ACTOR_KINDS
            and finite_number(item.collision_time_s)
            and item.collision_time_s > 0
            and finite_number(item.confidence)
            and finite_number(item.timestamp)
            and 0.65 <= item.confidence <= 1
            and 0 <= now - item.timestamp <= self.max_age
        ]
        if not valid:
            self.pulse_until = -math.inf
            return Decision(False, "WATCH", "no_fresh_path_conflict")
        closest = min(valid, key=lambda item: item.collision_time_s)
        ttc = closest.collision_time_s
        if ttc > 4:
            self.pulse_until = -math.inf
            return Decision(False, "WATCH", "outside_brake_horizon")
        if now < self.release_until:
            return Decision(False, "WATCH", "required_release_interval")
        if now >= self.pulse_until:
            maximum_pulse = min(
                0.10 if ttc < 1.5 else 0.06,
                (telemetry.speed_mps - stop_margin) / self.max_deceleration,
            )
            if maximum_pulse < 0.02:
                return Decision(False, "STOPPED", "insufficient_forward_speed_for_pulse")
            self.pulse_until = now + maximum_pulse
        # The gap is scheduled when an active pulse ends, independent of hazard jitter.
        return Decision(True, "BRAKING", "fresh_imminent_path_conflict")

    def tick(self, now, telemetry, hazards):
        if not finite_number(now):
            return self.fault("invalid_control_clock")
        if self.pulse_until != -math.inf and now + 1e-9 >= self.pulse_until:
            self.release_until = max(self.release_until, now + 0.04)
            self.pulse_until = -math.inf
        decision = self.decide(now, telemetry, hazards)
        if decision.s_pressed:
            decision = replace(decision, release_by=min(self.pulse_until, now + 0.1))
        return decision


class DryRunSink:
    def __init__(self):
        self.events = []
        self.pressed = False
        self.release_by = -math.inf

    def advance(self, timestamp):
        if self.pressed and timestamp + 1e-9 >= self.release_by:
            self.pressed = False
            self.events.append(
                dict(timestamp=timestamp, s_pressed=False, state="WATCHDOG", reason="lease_expired")
            )

    def apply(self, timestamp, decision):
        self.advance(timestamp)
        self.pressed = bool(
            decision.s_pressed
            and decision.release_by is not None
            and timestamp < decision.release_by <= timestamp + 0.100001
        )
        self.release_by = decision.release_by if self.pressed else -math.inf
        self.events.append(
            dict(
                timestamp=timestamp,
                s_pressed=self.pressed,
                state=decision.state,
                reason=decision.reason,
                release_by=decision.release_by,
            )
        )
