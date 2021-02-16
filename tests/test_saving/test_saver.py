import unittest
import torch
import sys

from modeling.train import Trainer
from ..helpers import config, std_objects
config = config.TestsConfig()


class TestSaver(unittest.TestCase):
    # TODO: let the assistant class make @from_model() classmethod
    def test_save_then_load(self):
        raise NotImplementedError

    def test_load_multiple_model(self):
        raise NotImplementedError("Not written yet. Take manager's tests as base")
