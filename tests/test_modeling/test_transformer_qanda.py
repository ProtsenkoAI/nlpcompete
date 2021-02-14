import unittest
import torch


from modeling.transformer_qanda import TransformerQA
from modeling.evaluating import Validator
from data.contain import DataContainer
from data.loaders_creation import DataLoaderSepXYCreator

from data.datasets import TrainDataset, EvalDataset

from . import weights_helpers
from ..helpers import config, std_objects
config = config.TestsConfig()

class TestTransformerQA(unittest.TestCase):
    def setUp(self):     
        self.model = std_objects.get_model()
        self.loader = std_objects.get_loader()

    def test_forward(self):
        features, (start_labels, end_labels) = next(iter(self.loader))
        start_logits, end_logits = self.model(*features)
        self.assertEqual(len(start_labels), len(start_logits))

    def test_reset_weights(self):
        transformer_weights_before = weights_helpers.get_weights(self.model.get_transformer())
        head_weights_before = weights_helpers.get_weights(self.model.get_head())

        self.model.reset_weights()
        transformer_weights_after = weights_helpers.get_weights(self.model.get_transformer())
        head_weights_after = weights_helpers.get_weights(self.model.get_head())

        res_head = weights_helpers.check_weights_equal(head_weights_before, head_weights_after)
        res_trans = weights_helpers.check_weights_equal(transformer_weights_before, transformer_weights_after)

        self.assertFalse(res_head, "head weights must change after reset")
        self.assertTrue(res_trans, "transformer weights should be equal after reset")

    def test_predict_start_end_idx(self):
        features, labels = next(iter(self.loader))

        start_idx, end_idx = self.model.predict_start_end_idx(*features)
        self.assertIsInstance(start_idx, int)
        self.assertIsInstance(end_idx, int)
