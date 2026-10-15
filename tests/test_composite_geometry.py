"""Ground-truth boxes must follow the labelled actor, not surrounding context."""
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from replay_composites import render


class CompositeGeometryTests(unittest.TestCase):
    def test_rider_context_does_not_enlarge_bicycle_ground_truth(self):
        background = Image.new("RGB", (960, 544))
        actor = Image.new("RGBA", (6, 8), (255, 0, 0, 255))
        actor.info["target_box"] = (0, 0.5, 1, 0.5)
        _frame, box = render(background, actor, "bicycle_crossing", 2)
        self.assertAlmostEqual(box[3] * 544, 54)
        self.assertAlmostEqual(box[2] * 960, 81)
        self.assertAlmostEqual((box[1] + box[3]) * 544, round(0.58 * 544))

    def test_no_inserted_actor_has_no_ground_truth_box(self):
        background = Image.new("RGB", (960, 544))
        actor = Image.new("RGBA", (6, 8), (255, 0, 0, 255))
        _frame, box = render(background, actor, "no_inserted_actor", 2)
        self.assertIsNone(box)


if __name__ == "__main__":
    unittest.main()
