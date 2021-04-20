import unittest
import torch

from sources.pipeline.modeling import ModelManager
from tests.helpers import config, std_objects, weights_helpers

config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.model = std_objects.get_model()
        self.mod_manager = std_objects.get_model_manager(self.model)

        self.batch_size = 4
        self.train_features = (["Тасс бла бла бла какая-то новость."] * self.batch_size,
                               ["бла бла бла @placeholder"] * self.batch_size
                              )
        self.labels = [0, 0, 0, 0, 1, 1, 1, 1]

        self.subm_featues = (["Тасс бла бла бла какая-то новость."] * self.batch_size,
                             ["бла бла бла @placeholder"] * self.batch_size,
                             list(range(5000, 5000 + self.batch_size)),
                             [202] * self.batch_size,
                             [210] * self.batch_size,
                             ["Исламское государство"] * self.batch_size
                              )


shared_objs = SharedObjects()


class TestModelManager(unittest.TestCase):
    def test_preproc_forward(self):
        proc_preds = shared_objs.mod_manager.preproc_forward(shared_objs.train_features)

        self.assertIsInstance(proc_preds, torch.Tensor)
        self.assertEqual(len(shared_objs.train_features[0]), len(proc_preds))

    def test_preproc_forward_labeled(self):
        batch = (shared_objs.train_features, shared_objs.labels)
        proc_preds, proc_labels = shared_objs.mod_manager.preproc_forward(*batch)
        print("proc_preds:", proc_preds)
        self.assertIsInstance(proc_preds, torch.Tensor)
        self.assertIsInstance(proc_labels, torch.Tensor)

    def test_reset_weights(self):
        old_weights = weights_helpers.get_weights(shared_objs.mod_manager.get_model())
        shared_objs.mod_manager.reset_model_weights()
        new_weights = weights_helpers.get_weights(shared_objs.mod_manager.get_model())
        self.assertFalse(weights_helpers.check_weights_equal(old_weights, new_weights))

    def test_predict_postproc(self):
        postproced = shared_objs.mod_manager.predict_postproc(
            shared_objs.subm_featues
        )
        self.assertEqual(len(postproced[0]), shared_objs.batch_size)
        self.assertEqual(len(postproced), 5) #check rucos_types for description

    def test_save_then_load(self):
        saver = std_objects.get_local_saver()
        model_name = shared_objs.mod_manager.save_model(saver)
        del shared_objs.model
        self.assertIsInstance(model_name, str)
        del shared_objs.mod_manager
        loaded_manager = ModelManager.load_checkpoint(saver, model_name)
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

        cuda_pred = manager.preproc_forward(shared_objs.train_features)

        self.assertTrue(cuda_pred[0].is_cuda)
