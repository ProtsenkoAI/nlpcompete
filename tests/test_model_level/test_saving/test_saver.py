import unittest

from tests.helpers import config

config = config.TestsConfig()


class TestSaver(unittest.TestCase):
    # TODO: let the assistant class make @from_model() classmethod
    def test_save_then_load(self):
        raise NotImplementedError

    def test_load_multiple_model(self):
        raise NotImplementedError("Not written yet. Take manager's tests as base")
