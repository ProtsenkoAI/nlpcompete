import unittest
import torch

from ..helpers import config, std_objects

config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.processor = std_objects.get_unbatching_processor()


shared_objs = SharedObjects()


class TestBlendingModelManager(unittest.TestCase):
    def test_preproc_forward(self):
        a1 = 0.2
        a2 = 0.1
        a3 = 0.25
        a4 = 0.75
        models_preds = [
            (
                torch.tensor([
                    [a1] * 16 for _ in range(8)
                ]),
                torch.tensor([
                    [a2] * 16 for _ in range(8)
                ])
            ),
            (
                torch.tensor([
                    [a3] * 16 for _ in range(8)
                ]),
                torch.tensor([
                    [a4] * 16 for _ in range(8)
                ])
            )
        ]
        k1 = 0.3
        k2 = 0.7
        blending_mm = std_objects.get_blending_model_manager(weights=[k1, k2], processor=shared_objs.processor)
        bruh = list(blending_mm.preproc_forward(models_preds))
        self.assertEqual(len(bruh), 8)
        self.assertEqual(bruh[0][0].shape, (512, ))
        self.assertEqual(bruh[0][1].shape, (512, ))
        self.assertAlmostEqual(bruh[0][0][0].item(), a1 * k1 + a3 * k2)
        self.assertAlmostEqual(bruh[0][1][0].item(), a2 * k1 + a4 * k2)


if __name__ == "__main__":

    unittest.main()
