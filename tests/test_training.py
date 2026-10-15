"""Optional training extras; ordinary runtime CI can skip without torch."""

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
try:
    import torch
    from train_focused import crop_box, distillation_loss, focused_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional training dependencies not installed")
class TrainingTests(unittest.TestCase):
    def test_distillation_detaches_teacher_and_ignores_void_pixels(self):
        student = torch.randn(1, 19, 3, 3, requires_grad=True)
        teacher = torch.randn(1, 19, 3, 3, requires_grad=True)
        target = torch.full((1, 3, 3), 255, dtype=torch.long)
        target[:, 0, 0] = 11
        loss = distillation_loss(student, teacher, target)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[:, :, 1:, 1:].abs().sum()), 0)
        self.assertGreater(float(student.grad[:, :, 0, 0].abs().sum()), 0)

    def test_identical_soft_predictions_have_zero_distillation_loss(self):
        values = torch.randn(1, 19, 3, 3)
        self.assertAlmostEqual(
            float(distillation_loss(values, values, torch.zeros(1, 3, 3, dtype=torch.long))),
            0,
            places=6,
        )

    def test_loss_and_gradient_match_an_analytic_one_pixel_case(self):
        logits = torch.zeros(1, 19, 1, 1)
        logits[:, 18] = math.log(3)
        logits.requires_grad_()
        weights = torch.ones(19)
        weights[18] = 2
        loss = focused_loss(logits, torch.full((1, 1, 1), 18, dtype=torch.long), weights)
        probability = 1 / 7
        expected = (1 - probability) * 2 * math.log(7) + 0.35 * 0.4
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        loss.backward()
        derivative = (
            (
                2 * math.log(probability)
                - 2 * (1 - probability) / probability
                - 1.05 / (probability + 2) ** 2
            )
            * probability
            * (1 - probability)
        )
        self.assertAlmostEqual(float(logits.grad[0, 18, 0, 0]), derivative, places=6)

    def test_target_crop_stays_inside_frame_and_retains_target(self):
        mask = np.zeros((720, 1280), np.uint8)
        mask[100:200, 1200:1240] = 18
        retained = 0
        for seed in range(100):
            x0, y0, x1, y1 = crop_box(mask, random.Random(seed), (960, 544))
            self.assertTrue(0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720)
            retained += bool((mask[y0:y1, x0:x1] == 18).any())
        self.assertGreaterEqual(retained, 70)

    def test_ignored_pixels_have_zero_gradient(self):
        logits = torch.randn(1, 19, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), 255, dtype=torch.long)
        target[:, 0, 0] = 18
        loss = focused_loss(logits, target, torch.ones(19))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[:, :, 1:, 1:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :, 0, 0].abs().sum()), 0)

    def test_all_ignored_batch_is_finite_and_differentiable(self):
        logits = torch.randn(1, 19, 4, 4, requires_grad=True)
        loss = focused_loss(logits, torch.full((1, 4, 4), 255, dtype=torch.long), torch.ones(19))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_correct_focus_prediction_has_smaller_loss_than_wrong_class(self):
        target = torch.full((1, 4, 4), 18, dtype=torch.long)
        correct = torch.zeros(1, 19, 4, 4)
        wrong = correct.clone()
        correct[:, 18] = 8
        wrong[:, 0] = 8
        self.assertLess(
            float(focused_loss(correct, target, torch.ones(19))),
            float(focused_loss(wrong, target, torch.ones(19))),
        )


if __name__ == "__main__":
    unittest.main()
