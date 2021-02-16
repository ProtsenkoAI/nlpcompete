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
        self.mod_manager = std_objects.get_model_manager(self.model)
        self.val_dataset = std_objects.get_val_dataset()
        self.val_loader = std_objects.get_loader(self.val_dataset)

shared_objs = SharedObjects()

class TestValidator(unittest.TestCase):
    def test_evaluate_one_metric(self):
        raise NotImplementedError

    def test_evaluate_many_metrics_get_report(self):
        raise NotImplementedError
