"""Offline protocol and pixel-mask checks; no training or inference."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Autonomous-Driving-App-GTA5"))
sys.path.insert(0, str(ROOT / "training"))
from autonomy.ego import polygon_mask, validate_polygon
from autonomy.perception import components
from group_split import partition


class AuditProtocolTests(unittest.TestCase):
    def test_guard_band_separates_neighboring_kept_ids(self):
        kept = [(i, partition(i)[0]) for i in range(10000) if partition(i)[0] != "excluded"]
        for (left, a), (right, b) in zip(kept, kept[1:], strict=False):
            if a != b:
                self.assertGreater(right - left, 20)
        self.assertEqual(partition(199)[0], "excluded")
        self.assertEqual(partition(200)[0], "excluded")

    def test_expanded_manifest_is_not_legacy_block_split(self):
        data = json.loads((ROOT / "experiments/expanded/manifest.json").read_text())
        self.assertIn("Official partitions", data["split_protocol"])
        self.assertEqual(
            data["split_sha256"], "d373bd9ad14bd5e551543441aa94ec4d16898f3d414e7f12abcc937cc55f2a0d"
        )
        self.assertEqual(
            len([r for r in data["records"] if r.get("split") == "test" and "excluded" not in r]),
            798,
        )

    def test_ego_mask_filters_vehicle_pixels_and_preserves_people_bicycles(self):
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[80:100, 10:90] = 13
        labels[84:94, 15:25] = 11
        labels[84:94, 30:40] = 18
        labels[20:30, 50:60] = 13
        confidence = np.ones(labels.shape, dtype=np.float32)
        ego = polygon_mask([[0, 0.75], [1, 0.75], [1, 1], [0, 1]], 100, 100)
        before = components(labels, confidence)
        after = components(labels, confidence, ego_mask=ego)
        self.assertEqual(sum(d["label"] == "car" for d in before), 2)
        self.assertEqual(sum(d["label"] == "car" for d in after), 1)
        self.assertEqual({d["label"] for d in after}, {"car", "person", "bicycle"})
        self.assertEqual(int((labels == 13).sum()), 1500)  # raw segmentation is not changed

    def test_invalid_ego_polygons_fail_closed(self):
        for points in (
            [],
            [[0, 0]] * 3,
            [[0, 0], [1, 0], [float("nan"), 1]],
            [[0, 0], [2, 0], [1, 1]],
        ):
            with self.assertRaises(ValueError):
                validate_polygon(points)
