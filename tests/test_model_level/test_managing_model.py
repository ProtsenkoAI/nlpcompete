import unittest
import torch

from model_level.managing_model import ModelManager
from ..helpers import config, std_objects, weights_helpers

config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.model = std_objects.get_model()
        self.mod_manager = std_objects.get_model_manager(self.model)

        self.batch_size = 4
        self.features = (["lul it's context"] * self.batch_size,
                         ["some question?"] * self.batch_size
                         )
        self.labels = ([2] * self.batch_size,
                       [6] * self.batch_size
                       )


shared_objs = SharedObjects()


class TestModelManager(unittest.TestCase):
    def test_preproc_forward(self):
        start_preds, end_preds = shared_objs.mod_manager.preproc_forward(shared_objs.features)

        self.assertIsInstance(start_preds, torch.Tensor)
        self.assertIsInstance(end_preds, torch.Tensor)
        self.assertEqual(len(shared_objs.features[0]), len(start_preds))

    def test_preproc_forward_labeled(self):
        batch = (shared_objs.features, shared_objs.labels)
        _, (proc_start_idxs, proc_end_idxs) = shared_objs.mod_manager.preproc_forward(*batch)

        self.assertIsInstance(proc_start_idxs, torch.Tensor)
        self.assertIsInstance(proc_end_idxs, torch.Tensor)

    def test_reset_weights(self):
        old_weights = weights_helpers.get_weights(shared_objs.mod_manager.get_model())
        shared_objs.mod_manager.reset_model_weights()
        new_weights = weights_helpers.get_weights(shared_objs.mod_manager.get_model())
        self.assertFalse(weights_helpers.check_weights_equal(old_weights, new_weights))

    def test_predict_postproc(self):
        answer_start_end_idxs = shared_objs.mod_manager.predict_postproc(
            shared_objs.features
        )
        self.assertEqual(len(answer_start_end_idxs), shared_objs.batch_size)

    def test_predict_get_text_labeled(self):
        batch = (shared_objs.features, shared_objs.labels)
        pred_texts, true_texts = shared_objs.mod_manager.predict_postproc(*batch)
        self.assertEqual(len(pred_texts), shared_objs.batch_size)
        self.assertEqual(len(true_texts), shared_objs.batch_size)

        self.assertIsInstance(pred_texts[0], str)
        self.assertIsInstance(true_texts[0], str)

    def test_save_then_load(self):
        saver = std_objects.get_local_saver()
        model_name = shared_objs.mod_manager.save_model(saver)
        del shared_objs.model
        self.assertIsInstance(model_name, str)
        del shared_objs.mod_manager
        loaded_manager = ModelManager.load(saver, model_name)
        self.assertIsInstance(loaded_manager, ModelManager)
        shared_objs.model = std_objects.get_model()
        shared_objs.mod_manager = std_objects.get_model_manager(model=shared_objs.model)

    @unittest.skipIf(not torch.cuda.is_available(),
                     "Cuda is not avalilable, skipping tests with it")
    def test_use_cuda(self):
        cuda_device = torch.device("cuda")
        del shared_objs.mod_manager
        manager = std_objects.get_model_manager(device=cuda_device)
        shared_objs.mod_manager = manager

        cuda_pred = manager.preproc_forward(shared_objs.features)

        self.assertTrue(cuda_pred[0].is_cuda)
