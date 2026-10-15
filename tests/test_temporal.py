import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from autonomy.control import BrakeController, Telemetry
from autonomy.geometry import CameraCalibration
from autonomy.hazards import estimate_hazards
from autonomy.tracking import Tracker
from replay_control import replay


class TemporalTests(unittest.TestCase):
    def test_lateral_velocity_remains_available_at_60_and_240_fps(self):
        for fps in (60, 240):
            tracker = Tracker()
            for frame in range(20):
                now = frame / fps
                track = tracker.update(
                    [dict(label="car", confidence=0.9, box=(0.2 + 0.2 * now, 0.4, 0.1, 0.2))], now
                )[0]
            self.assertAlmostEqual(track.lateral_velocity(), 0.2, places=5)

    def test_numpy_detection_boxes_produce_serializable_hazards(self):
        import json

        import numpy as np

        tracker = Tracker()
        tracks = tracker.update(
            [
                dict(
                    label="car",
                    confidence=0.9,
                    box=tuple(np.float64(x) for x in (0.4, 0.5, 0.1, 0.2)),
                )
            ],
            1,
        )
        hazards, facts = estimate_hazards(
            tracks, Telemetry(12, 1, True, False), CameraCalibration(1.5, 60, 0.4), 1
        )
        self.assertIs(hazards[0].path_conflict, True)
        json.dumps(facts)

    def test_replay_requires_explicit_gear_and_driver_status(self):
        for telemetry in (
            [dict(timestamp=0, speed_mps=12)],
            [dict(timestamp=math.nan, speed_mps=12, forward_gear=True, manual_override=False)],
        ):
            with self.assertRaises(ValueError):
                list(
                    replay(
                        [dict(media_time_s=0, detections=[])],
                        telemetry,
                        CameraCalibration(1.5, 60, 0.4),
                        True,
                    )
                )

    def test_lateral_prediction_keeps_id_during_faster_cut_in(self):
        tracker = Tracker()
        ids = []
        for now, x in ((0, 0.12), (0.05, 0.14), (0.10, 0.17), (0.20, 0.23)):
            track = tracker.update(
                [dict(label="car", confidence=0.9, box=(x, 0.5, 0.08, 0.2))], now
            )[0]
            ids.append(track.identifier)
        self.assertEqual(len(set(ids)), 1)
        self.assertGreater(track.lateral_velocity(), 0)

    def test_sidewalk_person_does_not_generate_path_conflict(self):
        tracker = Tracker()
        tracks = tracker.update(
            [dict(label="person", confidence=0.95, box=(0.02, 0.5, 0.04, 0.2))], 1
        )
        hazards, facts = estimate_hazards(
            tracks,
            Telemetry(12, 1, forward_gear=True, manual_override=False),
            CameraCalibration(1.5, 60, 0.4),
            1,
        )
        self.assertFalse(hazards[0].path_conflict)
        self.assertFalse(facts[0]["path_conflict"])

    def test_crossing_bicycle_is_predicted_before_current_lane_overlap(self):
        tracker = Tracker()
        for i in range(4):
            now = i * 0.05
            tracks = tracker.update(
                [
                    dict(
                        label="bicycle",
                        confidence=0.95,
                        box=(0.08 + i * 0.01, 0.3 + i * 0.001, 0.05, 0.2),
                    )
                ],
                now,
            )
        hazards, _ = estimate_hazards(
            tracks,
            Telemetry(12, now, forward_gear=True, manual_override=False),
            CameraCalibration(1.5, 60, 0.4),
            now,
        )
        self.assertTrue(hazards[0].path_conflict)

    def test_missing_or_stale_speed_and_camera_do_not_invent_distance(self):
        for camera, telemetry in (
            (None, Telemetry(12, 1, forward_gear=True, manual_override=False)),
            (CameraCalibration(1.5, 60, 0.4), None),
            (
                CameraCalibration(1.5, 60, 0.4),
                Telemetry(12, 0, forward_gear=True, manual_override=False),
            ),
        ):
            hazards, _ = estimate_hazards([], telemetry, camera, 1)
            self.assertEqual(hazards, [])

    def test_invalid_detection_and_clock_do_not_reach_hazard_estimator(self):
        tracker = Tracker()
        self.assertEqual(
            tracker.update([dict(label="car", confidence=0.9, box=(0, 0, math.nan, 1))], 1), []
        )
        with self.assertRaises(ValueError):
            tracker.update([], math.nan)

    def test_first_frame_stopped_car_has_explicit_assumption_and_can_brake(self):
        tracker = Tracker()
        tracks = tracker.update([dict(label="car", confidence=0.95, box=(0.4, 0.5, 0.15, 0.3))], 1)
        hazards, facts = estimate_hazards(
            tracks,
            Telemetry(12, 1, forward_gear=True, manual_override=False),
            CameraCalibration(1.5, 60, 0.4),
            1,
        )
        self.assertEqual(facts[0]["method"], "stationary_target_assumption")
        controller = BrakeController()
        controller.arm()
        self.assertTrue(
            controller.tick(
                1, Telemetry(12, 1, forward_gear=True, manual_override=False), hazards
            ).s_pressed
        )

    def test_trace_to_dry_brake_respects_manual_override_and_emits_no_input(self):
        observations = [
            dict(
                media_time_s=i * 0.05,
                detections=[dict(label="car", confidence=0.95, box=(0.4, 0.5, 0.15, 0.3))],
            )
            for i in range(5)
        ]
        telemetry = [
            dict(timestamp=i * 0.05, speed_mps=12, forward_gear=True, manual_override=i >= 2)
            for i in range(5)
        ]
        records = list(replay(observations, telemetry, CameraCalibration(1.5, 60, 0.4), arm=True))
        self.assertTrue(records[0]["simulated_s_pressed"])
        self.assertTrue(all(not x["simulated_s_pressed"] for x in records[2:]))
        self.assertTrue(all(x["game_input_emitted"] is False for x in records))

    def test_replay_rejects_nonincreasing_observation_clock(self):
        with self.assertRaises(ValueError):
            list(
                replay(
                    [dict(media_time_s=1), dict(media_time_s=1)],
                    [],
                    CameraCalibration(1.5, 60, 0.4),
                )
            )


if __name__ == "__main__":
    unittest.main()
