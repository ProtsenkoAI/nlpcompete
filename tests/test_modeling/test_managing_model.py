import unittest
import torch

from modeling.managing_model import ModelManager
from ..helpers import config, std_objects
from . import weights_helpers
config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.model = std_objects.get_model()
        self.mod_manager = std_objects.get_model_manager(self.model)

        # loader = std_objects.get_loader()
        # self.features, self.labels = next(iter(loader))
        batch_size = 4
        self.features = (["lul it's context"] * batch_size,
                         ["some question?"] * batch_size
                         )
        self.labels = ([101] * batch_size,
                       [202] * batch_size
                       )

shared_objs = SharedObjects()

class TestModelManager(unittest.TestCase):
    def test_preproc_forward(self):
        start_preds, end_preds = shared_objs.mod_manager.preproc_forward(shared_objs.features)

        self.assertIsInstance(start_preds, torch.Tensor)
        self.assertIsInstance(end_preds, torch.Tensor)
        self.assertEqual(len(shared_objs.features[0]), len(start_preds))

    def test_preproc_labels(self):
        proc_start_idxs, proc_end_idxs = shared_objs.mod_manager.preproc_labels(shared_objs.labels)

        self.assertIsInstance(proc_start_idxs, torch.Tensor)
        self.assertIsInstance(proc_end_idxs, torch.Tensor)

    def test_reset_weights(self):
        shared_objs.mod_manager.reset_model_weights()

    def test_predict_postproc(self):
        answer_start_end_idxs = shared_objs.mod_manager.predict_postproc(
                                                        shared_objs.features
                                                        )
        start_idx, end_idx = answer_start_end_idxs
        self.assertIsInstance(start_idx, int)
        self.assertIsInstance(end_idx, int)

    def test_save_then_load(self):
        model_name = shared_objs.mod_manager.save_model()
        self.assertIsInstance(model_name, str)
        loaded_model = ModelManager.load_model(model_name)

        src_model_type = type(shared_objs.model)
        self.assertIsInstance(loaded_model, src_model_type)

    def test_save_load_two_models_check_models_do_not_mix(self):
        src_model1 = shared_objs.model
        model_name1 = shared_objs.mod_manager.save_model()
        self.assertIsInstance(model_name1, str)
        loaded_model1 = ModelManager.load_model(model_name1)

        src_model2 = std_objects.get_model(head_nneurons=3)
        mod_manager2 = std_objects.get_model_manager(model=src_model2)
        model_name2 = mod_manager2.save_model()
        self.assertIsInstance(model_name2, str)
        loaded_model2 = ModelManager.load_model(model_name2)

        self.assertTrue(self._check_models_weights_equal(src_model1, loaded_model1))
        self.assertTrue(self._check_models_weights_equal(src_model2, loaded_model2))
        self.assertFalse(self._check_models_weights_equal(loaded_model1, loaded_model2))


    def _check_models_weights_equal(self, model1, model2):
        weights1 = weights_helpers.get_weights(model1)
        weights2 = weights_helpers.get_weights(model2)
        weights_are_equal = weights_helpers.check_weights_equal(weights1,
                                                                weights2)
        return weights_are_equal
