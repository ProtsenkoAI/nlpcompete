import unittest
import torch

from ..helpers import config, std_objects

config = config.TestsConfig()


class TestBlendingModelManager(unittest.TestCase):

    def test_preproc_forward(self):
        a1 = 3
        a2 = 1
        a3 = 1
        a4 = 5
        a5 = 0.6
        a6 = 0.5
        models_preds = [
            (
                torch.tensor([
                    [a1] * 512 for _ in range(8)
                ]),
                torch.tensor([
                    [a2] * 512 for _ in range(8)
                ])
            ),
            (
                torch.tensor([
                    [a3] * 512 for _ in range(8)
                ]),
                torch.tensor([
                    [a4] * 512 for _ in range(8)
                ])
            ),
            (
                torch.tensor([
                    [a5] * 512 for _ in range(8)
                ]),
                torch.tensor([
                    [a6] * 512 for _ in range(8)
                ])
            )
        ]
        k1 = 0.25
        k2 = 0.25
        k3 = 0.5
        blending_mm = std_objects.get_blending_model_manager(weights=[k1, k2, k3])
        start_probs, end_probs = blending_mm.preproc_forward(models_preds)
        self.assertEqual(start_probs.shape, models_preds[0][0].shape)
        self.assertEqual(end_probs.shape, models_preds[0][1].shape)
        self.assertAlmostEqual(start_probs[0][0].item(), a1 * k1 + a3 * k2 + a5 * k3)
        self.assertAlmostEqual(end_probs[0][0].item(), a2 * k1 + a4 * k2 + a6 * k3)


if __name__ == "__main__":

    unittest.main()
