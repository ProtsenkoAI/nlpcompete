import unittest
import torch
import sys

from modeling.train import Trainer
from ..helpers import config, std_objects
from . import weights_helpers
config = config.TestsConfig()

class SharedObjects:
    """The class is used to speed up testing"""
    def __init__(self):
        self.model = std_objects.get_model()
        self.trainer = std_objects.get_trainer(max_step=2, model=self.model, eval_steps=1, accum_iters=1)

shared_objs = SharedObjects()

class TestTrainer(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(shared_objs.trainer, Trainer)

    def test_fit(self):
        train = std_objects.get_train_dataset(nrows=1)
        val = std_objects.get_val_dataset(nrows=1)
        
        train_loader = std_objects.get_loader(train)
        val_loader = std_objects.get_loader(val)

        old_weights = weights_helpers.get_weights(shared_objs.model)
        shared_objs.trainer.fit(train_loader, val_loader)
        eval_vals = shared_objs.trainer.get_eval_vals()
        self.assertGreater(eval_vals, 0, "validation was not conducted during training")
        new_weights = weights_helpers.get_weights(shared_objs.model)

        weights_equal = weights_helpers.check_weights_equal(old_weights, new_weights)
        self.assertFalse(weights_equal, "training doesn't affect weights")
