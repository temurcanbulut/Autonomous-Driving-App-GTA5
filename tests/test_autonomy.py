import math
import sys
import unittest
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
from autonomy.capture import FrameProducer
from autonomy.control import BrakeController, DryRunSink, Hazard, Telemetry
from autonomy.geometry import CameraCalibration, corridor_conflict
from autonomy.perception import EXPECTED_CLASSES, LatestFrame, components, validate_metadata
from autonomy.tracking import Tracker


class ControlTests(unittest.TestCase):
    def test_unknown_gear_or_driver_status_and_bad_numeric_types_disarm(self):
        for telemetry in (
            Telemetry(12, 1),
            Telemetry(12, 1, True, None),
            Telemetry(12, 1, None, False),
            Telemetry(12, 1, "forward", False),
            Telemetry("12", 1, True, False),
            Telemetry(12, False, True, False),
        ):
            controller = self.controller()
            self.assertFalse(controller.tick(1, telemetry, [Hazard("car", 1, 0.9, 1)]).s_pressed)
            self.assertFalse(controller.armed)

    def test_invalid_clock_after_braking_releases_and_disarms(self):
        controller = self.controller()
        telemetry = Telemetry(12, 1, True, False)
        self.assertTrue(controller.tick(1, telemetry, [Hazard("car", 1, 0.9, 1)]).s_pressed)
        self.assertFalse(controller.tick("bad", telemetry, []).s_pressed)
        self.assertFalse(controller.armed)

    def test_unrecognized_actor_or_bad_hazard_timestamp_cannot_brake(self):
        for hazard in (
            Hazard("traffic_light", 1, 0.9, 1),
            Hazard("person", 1, 0.9, "bad"),
            Hazard("car", 1, 0.9, 1, path_conflict="yes"),
        ):
            self.assertFalse(
                self.controller().tick(1, Telemetry(12, 1, True, False), [hazard]).s_pressed
            )

    def controller(self):
        control = BrakeController()
        control.arm()
        return control

    def test_starts_disarmed_and_has_no_throttle_command(self):
        decision = BrakeController().tick(
            1,
            Telemetry(10, 1, forward_gear=True, manual_override=False),
            [Hazard("person", 1, 0.9, 1)],
        )
        self.assertFalse(decision.s_pressed)
        self.assertFalse(decision.throttle)

    def test_fresh_crossing_hazard_brakes(self):
        self.assertTrue(
            self.controller()
            .tick(
                1,
                Telemetry(12, 1, forward_gear=True, manual_override=False),
                [Hazard("person", 1.2, 0.9, 1)],
            )
            .s_pressed
        )

    def test_clear_path_and_sidewalk_objects_do_not_brake(self):
        for hazards in (
            [],
            [Hazard("person", 1, 0.9, 1, path_conflict=False)],
            [Hazard("car", 5, 0.9, 1)],
        ):
            self.assertFalse(
                self.controller()
                .tick(1, Telemetry(12, 1, forward_gear=True, manual_override=False), hazards)
                .s_pressed
            )

    def test_missing_invalid_stale_or_reverse_telemetry_latches_disarm(self):
        for telemetry in (
            None,
            Telemetry(12, 0, forward_gear=True, manual_override=False),
            Telemetry(12, 2, forward_gear=True, manual_override=False),
            Telemetry(-1, 1, forward_gear=True, manual_override=False),
            Telemetry(math.nan, 1, forward_gear=True, manual_override=False),
            Telemetry(12, 1, forward_gear=False, manual_override=False),
        ):
            controller = self.controller()
            decision = controller.tick(1, telemetry, [Hazard("car", 1, 0.9, 1)])
            self.assertFalse(decision.s_pressed)
            self.assertFalse(controller.armed)

    def test_manual_override_latches_disarm(self):
        controller = self.controller()
        self.assertFalse(
            controller.tick(
                1, Telemetry(12, 1, manual_override=True, forward_gear=True), []
            ).s_pressed
        )
        self.assertFalse(controller.armed)

    def test_low_speed_releases_before_reverse(self):
        for speed in (0, 0.1, 0.25):
            self.assertFalse(
                self.controller()
                .tick(
                    1,
                    Telemetry(speed, 1, forward_gear=True, manual_override=False),
                    [Hazard("person", 0.5, 1, 1)],
                )
                .s_pressed
            )

    def test_continuous_pulses_are_bounded(self):
        controller = self.controller()
        longest = current = 0
        saw_release = False
        for tick in range(200):
            now = tick * 0.02
            decision = controller.tick(
                now,
                Telemetry(20, now, forward_gear=True, manual_override=False),
                [Hazard("car", 1, 0.9, now)],
            )
            if decision.s_pressed:
                current += 0.02
                longest = max(current, longest)
            else:
                current = 0
                saw_release = True
        self.assertTrue(saw_release)
        self.assertLessEqual(longest, 0.10000001)

    def test_watchdog_expires_brake_lease_when_frames_stop(self):
        sink = DryRunSink()
        decision = self.controller().tick(
            1,
            Telemetry(12, 1, forward_gear=True, manual_override=False),
            [Hazard("person", 1, 0.9, 1)],
        )
        sink.apply(1, decision)
        self.assertTrue(sink.pressed)
        sink.advance(1.2)
        self.assertFalse(sink.pressed)

    def test_stale_low_confidence_and_nonfinite_hazards_do_not_brake(self):
        for hazard in (
            Hazard("person", 1, 0.9, 0),
            Hazard("person", 1, 0.2, 1),
            Hazard("person", math.nan, 0.9, 1),
            Hazard("person", 1, math.inf, 1),
            Hazard("person", 0, 0.9, 1),
        ):
            self.assertFalse(
                self.controller()
                .tick(1, Telemetry(10, 1, forward_gear=True, manual_override=False), [hazard])
                .s_pressed
            )

    @settings(max_examples=2000, derandomize=True, deadline=None)
    @given(
        st.floats(min_value=0, max_value=0.25, allow_nan=False),
        st.floats(min_value=0.01, max_value=3, allow_nan=False),
    )
    def test_arbitrary_near_stopped_states_never_press_reverse_key(self, speed, ttc):
        self.assertFalse(
            self.controller()
            .tick(
                1,
                Telemetry(speed, 1, forward_gear=True, manual_override=False),
                [Hazard("person", ttc, 1, 1)],
            )
            .s_pressed
        )


class PerceptionTests(unittest.TestCase):
    def test_capture_producer_terminates_and_surfaces_source_errors(self):
        def failed_source():
            raise ValueError("fixture failure")
            yield None

        producer = FrameProducer(failed_source())
        try:
            with self.assertRaises(RuntimeError):
                producer.get()
        finally:
            producer.close()

    def test_checkpoint_class_order_and_dimensions_are_validated(self):
        metadata = dict(
            model="lraspp_mobilenet_v3_large", classes=list(EXPECTED_CLASSES), width=768, height=432
        )
        validate_metadata(metadata)
        metadata["classes"][0], metadata["classes"][1] = (
            metadata["classes"][1],
            metadata["classes"][0],
        )
        with self.assertRaises(ValueError):
            validate_metadata(metadata)
        metadata["classes"] = list(EXPECTED_CLASSES)
        metadata["width"] = 0
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_components_are_classified_and_confidence_filtered(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[30:60, 40:50] = 11
        confidence = np.ones((100, 100), np.float32) * 0.9
        result = components(mask, confidence)
        self.assertEqual(result[0]["label"], "person")
        self.assertAlmostEqual(result[0]["box"][0], 0.4)
        self.assertEqual(components(mask, confidence * 0.1), [])

    def test_latest_frame_discards_old_work(self):
        slot = LatestFrame()
        slot.put("old", 1)
        slot.put("new", 2)
        self.assertEqual(slot.get(), ("new", 2))
        self.assertEqual(slot.dropped, 1)
        slot.close()
        self.assertIsNone(slot.get())

    def test_tracker_retains_id_and_drops_stale_hazards(self):
        tracker = Tracker()
        detection = dict(label="car", confidence=0.9, box=(0.4, 0.4, 0.2, 0.2))
        first = tracker.update([detection], 1)[0]
        second = tracker.update([detection], 1.1)[0]
        self.assertEqual(first.identifier, second.identifier)
        self.assertEqual(tracker.update([], 1.2), [])
        self.assertFalse(tracker.tracks == {})
        tracker.update([], 2)
        self.assertEqual(tracker.tracks, {})

    def test_expanding_track_has_optical_ttc(self):
        tracker = Tracker()
        for index, size in enumerate((0.1, 0.11, 0.12, 0.13)):
            track = tracker.update(
                [dict(label="car", confidence=0.9, box=(0.4, 0.4, size, size))], index * 0.1
            )[0]
        self.assertIsNotNone(track.optical_ttc())
        self.assertGreater(track.optical_ttc(), 0)


class GeometryTests(unittest.TestCase):
    def test_calibration_rejects_horizon_and_invalid_values(self):
        camera = CameraCalibration(1.5, 60, 0.4)
        self.assertIsNone(camera.distance(0.4))
        self.assertGreater(camera.distance(0.6), camera.distance(0.8))
        with self.assertRaises(ValueError):
            CameraCalibration(-1, 60, 0.4).distance(0.8)

    def test_lateral_crossing_projects_into_corridor(self):
        box = (0.05, 0.6, 0.05, 0.2)
        self.assertFalse(corridor_conflict(box))
        self.assertTrue(corridor_conflict(box, lateral_velocity=0.5, horizon_seconds=0.8))


if __name__ == "__main__":
    unittest.main()
