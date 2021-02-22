import unittest
import torch
import numpy as np

from ..helpers import config, std_objects

config = config.TestsConfig()


class SharedObjects:
    """The class is used to speed up testing"""
    def __init__(self):
        loader = std_objects.get_loader()
        self.features, self.labels = next(iter(loader))
        self.batch_size = len(self.features[0])
        self.processor = std_objects.get_qa_processor()


shared_objs = SharedObjects()


class TestQADataProcessor(unittest.TestCase):
    def test_preprocess_unlabeled(self):
        preproc_features = shared_objs.processor.preprocess(shared_objs.features)
        for feature_type in preproc_features:
            self.assertIsInstance(feature_type, torch.Tensor)
            self.assertEqual(len(feature_type), len(shared_objs.features[0]))

    def test_preprocess_labeled(self):
        _, preproc_labels = shared_objs.processor.preprocess(shared_objs.features, shared_objs.labels)
        for label_categ in preproc_labels:
            self.assertIsInstance(label_categ, torch.Tensor)
            self.assertEqual(len(label_categ), len(shared_objs.features[0]))

    def test_postprocess_unlabeled(self):
        preds = self._imitate_preds()
        postproc_preds = shared_objs.processor.postprocess(preds)
        self.assertIsInstance(postproc_preds, list)
        self.assertEqual(len(postproc_preds), shared_objs.batch_size)

    def test_postprocess_labeled(self):
        preds = self._imitate_preds()
        _, preproc_labels = shared_objs.processor.preprocess(shared_objs.features, shared_objs.labels)
        _, postproc_labels = shared_objs.processor.postprocess(preds, preproc_labels)
        self.assertEqual(shared_objs.batch_size, len(postproc_labels))
        self.assertEqual(2, len(postproc_labels[0]))

    def test_text_from_token_ids(self):
        preds = self._imitate_preds()
        _, preproc_labels = shared_objs.processor.preprocess(shared_objs.features, shared_objs.labels)
        postproc_preds, postproc_labels = shared_objs.processor.postprocess(preds, preproc_labels)

        ground_truth_text = shared_objs.processor.text_from_token_idxs(postproc_labels, shared_objs.features)
        pred_answers = shared_objs.processor.text_from_token_idxs(postproc_preds, shared_objs.features)
        for true_answer in ground_truth_text[:3]:
            self.assertIsInstance(true_answer, str)
        for predict in pred_answers[:3]:
            self.assertIsInstance(predict, str)

    def _imitate_preds(self):
        seq_len = 512
        start_logits = torch.rand(len(shared_objs.features[0]), seq_len)
        end_logits = torch.rand(len(shared_objs.features[0]), seq_len)
        return start_logits, end_logits
