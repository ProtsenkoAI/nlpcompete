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
        self.trainer = std_objects.get_trainer(max_step=2, model=self.model)

shared_objs = SharedObjects()

class TestTrainer(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(shared_objs.trainer, Trainer)

    def test_fit(self):
        train = std_objects.get_train_dataset()
        test = std_objects.get_test_dataset()
        
        train_loader = std_objects.get_loader(train)
        test_loader = std_objects.get_loader(test)

        old_weights = weights_helpers.get_weights(shared_objs.model)
        shared_objs.trainer.fit(train_loader, test_loader)
        new_weights = weights_helpers.get_weights(shared_objs.model)

        weights_equal = weights_helpers.check_weights_equal(old_weights, new_weights)
        self.assertFalse(weights_equal, "training doesn't affect weights")

