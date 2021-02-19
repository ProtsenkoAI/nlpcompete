import unittest
import torch

from . import weights_helpers
from ..helpers import config, std_objects
config = config.TestsConfig()


class SharedObjects:
    """The class is used to speed up testing"""
    def __init__(self):
        self.model = std_objects.get_model()
        loader = std_objects.get_loader()
        processor = std_objects.get_qa_processor()
        features_unproc, labels_unproc = next(iter(loader))
        # self.features = processor.preprocess_features(features_unproc)
        # self.labels = processor.preprocess_labels(labels_unproc)
        self.features, self.labels = processor.preprocess_features_and_labels(features_unproc, labels_unproc)


shared_objs = SharedObjects()


class TestTransformerQA(unittest.TestCase):
    def test_forward(self):
        start_logits, end_logits = shared_objs.model(shared_objs.features)
        start_labels, end_labels = shared_objs.labels
        self.assertEqual(len(start_labels), len(start_logits))
        self.assertGreaterEqual(start_logits.ndim, 2, "preds have to have at least 2 dims: batch and token")

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

