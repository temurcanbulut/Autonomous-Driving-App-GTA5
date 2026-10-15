import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Autonomous-Driving-App-GTA5"))
import vision


def row(class_id, score=0.9, box=(0.5, 0.5, 0.3, 0.3)):
    values = np.zeros(85, dtype=np.float32)
    values[:4] = box
    values[4] = 0.8
    values[class_id + 5] = score
    return values


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.transform = vision.Transform(1, 1, 0, 0, 416)

    def decode(self, rows, allowed=None):
        return vision.decode_detections(
            [np.array(rows, dtype=np.float32).reshape(-1, 85)],
            (416, 416, 3),
            self.transform,
            0.5,
            0.4,
            80,
            allowed,
        )

    def test_complete_coco_class_order(self):
        names = (vision.BASE / "coco.names").read_text().splitlines()
        self.assertEqual(len(names), 80)
        self.assertEqual(names[4], "airplane")
        self.assertEqual(names[5], "bus")
        self.assertEqual(names[7], "truck")
        self.assertEqual(names[9], "traffic light")

    def test_nms_only_suppresses_same_class(self):
        detections = self.decode([row(2, 0.9), row(2, 0.8), row(0, 0.85)])
        self.assertEqual([item.class_id for item in detections], [2, 0])

    def test_does_not_double_count_opencv_region_objectness(self):
        item = row(2, 0.6)
        item[4] = 0.6
        detections = self.decode([item])
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.6, places=5)

    def test_empty_low_confidence_and_class_filter(self):
        self.assertEqual(self.decode([]), [])
        self.assertEqual(self.decode([row(2, 0.4)]), [])
        detections = self.decode([row(4), row(5)], vision.TRAFFIC_CLASSES)
        self.assertEqual([item.class_id for item in detections], [5])

    def test_nonfinite_and_outside_boxes_rejected(self):
        invalid = row(2)
        invalid[0] = np.nan
        self.assertEqual(self.decode([invalid, row(2, box=(2, 2, 0.1, 0.1))]), [])

    def test_mismatched_output_rejected(self):
        with self.assertRaises(ValueError):
            vision.decode_detections([np.zeros((1, 14))], (416, 416), self.transform, 0.5, 0.4, 80)

    def test_letterbox_restores_widescreen_coordinates(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        image, transform = vision.letterbox(frame, 416)
        self.assertEqual(image.shape, (416, 416, 3))
        item = row(2, box=(0.5, 0.5, 0.1, 0.05625))
        detection = vision.decode_detections(
            [np.array([item])], frame.shape, transform, 0.5, 0.4, 80
        )[0]
        for actual, expected in zip(detection.box, (576, 324, 128, 72), strict=True):
            self.assertAlmostEqual(actual, expected, delta=2)

    def test_box_clips_to_frame(self):
        item = self.decode([row(2, box=(0.05, 0.05, 0.3, 0.3))])[0]
        self.assertEqual(item.box[:2], (0, 0))

    def test_nearest_is_pixel_offset_and_excludes_traffic_light(self):
        car = vision.Detection(2, 0.9, (40, 60, 20, 20))
        light = replace(car, class_id=9, box=(40, 80, 20, 20))
        item, offset = vision.nearest_in_image([car, light], (100, 100, 3))
        self.assertEqual(item, car)
        self.assertEqual(offset, 30)
        self.assertIsNone(vision.nearest_in_image([light], (100, 100, 3)))


class PipelineTests(unittest.TestCase):
    def test_rejects_short_label_file_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.names").write_text("car\n")
            (root / "weights").touch()
            args = vision.parser().parse_args(
                ["--names", str(root / "bad.names"), "--weights", str(root / "weights")]
            )
            with (
                patch("vision.cv2.dnn.readNetFromDarknet") as load,
                self.assertRaisesRegex(ValueError, "label file"),
            ):
                vision.Detector(args)
            load.assert_not_called()

    def test_missing_weights_has_actionable_error(self):
        args = vision.parser().parse_args(["--weights", "does-not-exist.weights"])
        with self.assertRaisesRegex(ValueError, "README"):
            vision.Detector(args)

    def test_video_export_and_metrics_without_game_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = vision.parser().parse_args(
                [
                    "--source",
                    "fixture.avi",
                    "--headless",
                    "--max-frames",
                    "2",
                    "--output",
                    str(root / "result.avi"),
                    "--metrics",
                    str(root / "metrics.json"),
                ]
            )
            detector = MagicMock(names=["person"] * 80)
            detector.detect.return_value = ([], 5, 6)

            def source(_args):
                for _ in range(3):
                    yield np.zeros((128, 192, 3), dtype=np.uint8)

            with (
                patch("vision.Detector", return_value=detector),
                patch("vision.frames", side_effect=source),
                patch("vision.cv2.imshow") as show,
            ):
                metrics = vision.run(args)
            show.assert_not_called()
            self.assertEqual(metrics["frames"], 2)
            self.assertEqual(metrics["mean_inference_ms"], 5)
            self.assertEqual(json.loads((root / "metrics.json").read_text())["frames"], 2)
            capture = cv2.VideoCapture(str(root / "result.avi"))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                capture.release()
            # Exercise the real file reader and verify release on early close.
            args.source = str(root / "result.avi")
            stream = vision.frames(args)
            self.assertEqual(next(stream).shape, (128, 192, 3))
            stream.close()


if __name__ == "__main__":
    unittest.main()
