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


class SharedObjects:
    """The class is used to speed up testing"""
    def __init__(self):
        self.model = std_objects.get_model()
        self.loader = std_objects.get_loader()

shared_objs = SharedObjects()


class TestTransformerQA(unittest.TestCase):
    def test_forward(self):
        features, (start_labels, end_labels) = next(iter(shared_objs.loader))
        start_logits, end_logits = shared_objs.model(features)
        self.assertEqual(len(start_labels), len(start_logits))

    def test_reset_weights(self):
        transformer_weights_before = weights_helpers.get_weights(shared_objs.model.get_transformer())
        head_weights_before = weights_helpers.get_weights(shared_objs.model.get_head())

        shared_objs.model.reset_weights()
        transformer_weights_after = weights_helpers.get_weights(shared_objs.model.get_transformer())
        head_weights_after = weights_helpers.get_weights(shared_objs.model.get_head())

        res_head = weights_helpers.check_weights_equal(head_weights_before, head_weights_after)
        res_trans = weights_helpers.check_weights_equal(transformer_weights_before, transformer_weights_after)

        self.assertFalse(res_head, "head weights must change after reset")
        self.assertTrue(res_trans, "transformer weights should be equal after reset")

