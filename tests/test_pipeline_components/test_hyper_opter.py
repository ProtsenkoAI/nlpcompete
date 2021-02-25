import unittest

from tests.helpers import config, std_objects
config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.pipeline = std_objects.get_fit_eval_pipeline()


shared_objs = SharedObjects()


class TestHyperOpter(unittest.TestCase):
    def test_run_optimize(self):
        hyper_opter = HyperOpter()