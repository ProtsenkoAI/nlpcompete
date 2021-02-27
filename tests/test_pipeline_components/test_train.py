import unittest

from pipeline_components.train import Trainer
from model_level.managing_model import ModelManager
from tests.helpers import config, std_objects, weights_helpers

config = config.TestsConfig()


class SharedObjects:
    """The class is used to speed up testing"""

    def __init__(self):
        self.model = std_objects.get_model()
        self.manager = std_objects.get_model_manager(self.model)
        self.trainer = std_objects.get_trainer(weights_updater_kwargs={"accum_iters": 3})


shared_objs = SharedObjects()


class TestTrainer(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(shared_objs.trainer, Trainer)

    def test_fit_and_eval(self):
        print("test_fit_and_eval")
        dataset = std_objects.get_train_dataset(nrows=2)
        train_loader, val_loader = std_objects.get_data_assistant().train_test_split(dataset, test_size=0.1)
        old_weights = weights_helpers.get_weights(shared_objs.model)
        shared_objs.trainer.fit(train_loader, val_loader, shared_objs.manager, max_step=2, steps_betw_evals=1)
        eval_vals = shared_objs.trainer.get_eval_vals()
        self.assertGreater(len(eval_vals), 0, "validation was not conducted during training")
        self.assertTrue(isinstance(eval_vals[0], (float, int)))
        new_weights = weights_helpers.get_weights(shared_objs.model)

        weights_equal = weights_helpers.check_weights_equal(old_weights, new_weights)
        self.assertFalse(weights_equal, "training doesn't affect weights")

        del shared_objs.model, shared_objs.manager
        best_manager = shared_objs.trainer.load_best_manager()
        shared_objs.manager = best_manager
        self.assertIsInstance(best_manager, ModelManager)

    def test_full_dataset(self):
        dataset = std_objects.get_train_dataset(nrows=20 * config.batch_size)
        train_loader, val_loader = std_objects.get_data_assistant().train_test_split(dataset, test_size=0.1)
        print("starting test_full_dataset")
        shared_objs.trainer.fit(train_loader, val_loader, shared_objs.manager, max_step=15, steps_betw_evals=10)
