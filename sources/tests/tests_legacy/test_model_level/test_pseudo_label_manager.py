# import unittest
# import torch
#
# from model_level.managing_model import ModelManager
# from ..helpers import config, std_objects, weights_helpers
#
# config = config.TestsConfig()
#
#
# class SharedObjects:
#     def __init__(self):
#         self.model = std_objects.get_model()
#         self.mod_manager = std_objects.get_pseudo_label_manager(self.model)
#
#         self.batch_size = 4
#         self.features = (["lul it's context"] * self.batch_size,
#                          ["some question?"] * self.batch_size
#                          )
#
#
# shared_objs = SharedObjects()
#
#
# class TestModelManager(unittest.TestCase):
#     def test_preproc_forward(self):
#         start_preds, end_preds = shared_objs.mod_manager.preproc_forward(shared_objs.features)
#
#         self.assertIsInstance(start_preds, torch.Tensor)
#         self.assertIsInstance(end_preds, torch.Tensor)
#         self.assertEqual(len(shared_objs.features[0]), len(start_preds))
#
#     def test_predict_postproc(self):
#         start_end_chars, probs = shared_objs.mod_manager.predict_postproc(
#             shared_objs.features
#         )
#         self.assertEqual(len(start_end_chars), shared_objs.batch_size)
#         self.assertEqual(probs.ndim, 1)
#         self.assertEqual(len(probs), shared_objs.batch_size)
