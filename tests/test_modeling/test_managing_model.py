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
        self.batch_size = 4
        self.features = (["lul it's context"] * self.batch_size,
                         ["some question?"] * self.batch_size
                         )
        self.labels = ([101] * self.batch_size,
                       [202] * self.batch_size
                           )

shared_objs = SharedObjects()

class TestModelManager(unittest.TestCase):
    def test_preproc_forward(self):
        start_preds, end_preds = shared_objs.mod_manager.preproc_forward(shared_objs.features)

        self.assertIsInstance(start_preds, torch.Tensor)
        self.assertIsInstance(end_preds, torch.Tensor)
        self.assertEqual(len(shared_objs.features[0]), len(start_preds))

    def test_preproc_forward_labeled(self):
        _, (proc_start_idxs, proc_end_idxs) = shared_objs.mod_manager.preproc_forward_labeled(shared_objs.features,
                                                                                         shared_objs.labels)

        self.assertIsInstance(proc_start_idxs, torch.Tensor)
        self.assertIsInstance(proc_end_idxs, torch.Tensor)

    def test_reset_weights(self):
        shared_objs.mod_manager.reset_model_weights()

    def test_predict_postproc(self):
        answer_start_end_idxs = shared_objs.mod_manager.predict_postproc(
                                                        shared_objs.features
                                                        )
        print("postproc res", answer_start_end_idxs)
        self.assertEqual(len(answer_start_end_idxs), shared_objs.batch_size)

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

        src_model1_weights = weights_helpers.get_weights(src_model1)
        loaded_model1_weights = weights_helpers.get_weights(loaded_model1)
        del src_model1, loaded_model1

        src_model2 = std_objects.get_model(head_nneurons=3)
        mod_manager2 = std_objects.get_model_manager(model=src_model2)
        model_name2 = mod_manager2.save_model()
        del mod_manager2
        self.assertIsInstance(model_name2, str)
        loaded_model2 = ModelManager.load_model(model_name2)

        src_model2_weights = weights_helpers.get_weights(src_model2)
        loaded_model2_weights = weights_helpers.get_weights(loaded_model2)
        del src_model2, loaded_model2

        self.assertTrue(weights_helpers.check_weights_equal(src_model1_weights, loaded_model1_weights))
        self.assertTrue(weights_helpers.check_weights_equal(src_model2_weights, loaded_model2_weights))
        self.assertFalse(weights_helpers.check_weights_equal(loaded_model1_weights, loaded_model2_weights))

    @unittest.skipIf(not torch.cuda.is_available(), 
                     "Cuda is not avalilable, skipping tests with it")
    def test_use_cuda(self):
        cuda_device = torch.device("cuda")
        del shared_objs.mod_manager
        manager = std_objects.get_model_manager(device=cuda_device)
        shared_objs.mod_manager = manager

        cuda_pred = manager.preproc_forward(shared_objs.features)

        self.assertTrue(cuda_pred[0].is_cuda)
