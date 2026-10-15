import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


metrics = module("training_metrics", ROOT / "training" / "metrics.py")
splits = module("group_split", ROOT / "training" / "group_split.py")


class ExperimentTests(unittest.TestCase):
    def test_metrics_match_hand_computed_confusion(self):
        report = metrics.summarize([[8, 2], [1, 9]], ["road", "person"])
        self.assertAlmostEqual(report["pixel_accuracy"], 0.85)
        self.assertAlmostEqual(report["mean_iou"], (8 / 11 + 9 / 12) / 2)
        self.assertAlmostEqual(report["classes"][1]["recall"], 0.9)

    def test_absent_class_is_undefined_not_perfect(self):
        report = metrics.summarize([[10, 0], [0, 0]], ["road", "bicycle"])
        self.assertIsNone(report["classes"][1]["iou"])
        self.assertIsNone(report["classes"][1]["recall"])
        self.assertEqual(report["classes"][1]["support_pixels"], 0)

    def test_empty_or_negative_confusion_is_invalid(self):
        for value in ([[0, 0], [0, 0]], [[-1, 0], [0, 1]], [[1]]):
            with self.assertRaises(ValueError):
                metrics.summarize(value, ["road", "person"])

    def test_frame_id_blocks_never_span_partitions(self):
        for group in range(250):
            assigned = {splits.partition(group * 100 + offset)[0] for offset in range(100)}
            self.assertEqual(len(assigned), 1)
        self.assertEqual(splits.partition(1234), splits.partition(1234))


if __name__ == "__main__":
    unittest.main()
