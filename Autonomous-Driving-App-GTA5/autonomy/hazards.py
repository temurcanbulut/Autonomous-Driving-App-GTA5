"""Image tracks to explicitly approximate hazards, for offline replay only."""

import statistics

from .control import Hazard, finite_number
from .geometry import corridor_conflict


def estimate_hazards(tracks, telemetry, camera, now, *, max_age=0.15):
    """Return dry-run hazards plus assumptions; never produces keyboard input.

    Distance assumes a flat plane and fixed calibrated camera. Initial tracks
    assume a stationary target until relative motion is observable. Segmentation
    components are not ground-truth object instances, and confidence is not a
    calibrated collision probability.
    """
    if telemetry is None or camera is None:
        return [], [dict(reason="calibrated_camera_and_speed_required")]
    if not all(finite_number(x) for x in (now, telemetry.timestamp, telemetry.speed_mps)):
        return [], [dict(reason="invalid_telemetry")]
    if telemetry.forward_gear is not True or telemetry.manual_override is not False:
        return [], [dict(reason="gear_or_driver_status_unconfirmed")]
    if not 0 <= now - telemetry.timestamp <= max_age or telemetry.speed_mps < 0:
        return [], [dict(reason="stale_or_reverse_telemetry")]
    hazards, observations = [], []
    for track in tracks:
        if not 0 <= now - track.timestamp <= max_age:
            continue
        distance = camera.distance(track.box[1] + track.box[3])
        if distance is None:
            continue
        history = [(t, camera.distance(box[1] + box[3])) for t, box in track.states]
        history = [(t, d) for t, d in history if d is not None]
        closing = telemetry.speed_mps
        method = "stationary_target_assumption"
        if len(history) >= 3 and history[-1][0] - history[0][0] >= 0.08:
            rates = [
                (d0 - d1) / (t1 - t0)
                for (t0, d0), (t1, d1) in zip(history, history[1:], strict=False)
                if t1 > t0
            ]
            if rates:
                closing = max(0.0, min(60.0, statistics.median(rates)))
                method = "flat_ground_plane_track_derivative"
        if closing <= 0.1:
            ttc = None
        else:
            ttc = max(0.001, distance - 4.0) / closing
        lateral = track.lateral_velocity()
        projection = min(2.0, ttc) if ttc is not None else 0
        conflict = bool(
            corridor_conflict(track.box)
            or any(
                corridor_conflict(track.box, lateral, max(0, projection + dt))
                for dt in (-0.25, 0, 0.25)
            )
        )
        observations.append(
            dict(
                id=track.identifier,
                label=track.label,
                distance_m_approx=distance,
                closing_speed_mps_approx=closing,
                collision_time_s_approx=ttc,
                lateral_velocity_image_s=lateral,
                path_conflict=conflict,
                method=method,
            )
        )
        if ttc is not None:
            hazards.append(
                Hazard(track.label, ttc, track.confidence, track.timestamp, bool(conflict))
            )
    return hazards, observations
