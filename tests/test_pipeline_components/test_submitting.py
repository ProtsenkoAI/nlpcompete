import unittest
import os

from tests.helpers import config, std_objects
from pipeline_components.submitting import Submitter
config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.manager = std_objects.get_model_manager()
        self.subm_dir = "../test_data/"
        self.submitter = Submitter(subm_dir=self.subm_dir)


shared_objs = SharedObjects()


class TestSubmitter(unittest.TestCase):
    def test_make_submission_correct_dataset(self):
        test_dataset = std_objects.get_subm_dataset()
        loader = std_objects.get_loader(test_dataset, has_answers=False)
        shared_objs.submitter.create_submission(shared_objs.manager, loader, subm_file_name="subm")
        subm_path = os.path.join(shared_objs.subm_dir, "subm.json")
        file_saved = os.path.isfile(subm_path)
        self.assertTrue(file_saved)

    def test_make_submission_invalid_dataset(self):
        train_dataset = std_objects.get_subm_dataset()
        loader = std_objects.get_loader(train_dataset)
        with self.assertRaises(Exception):
            shared_objs.submitter.create_submission(shared_objs.manager, loader, subm_file_name="subm_failed")
        subm_path = os.path.join(shared_objs.subm_dir, "subm_failed.json")
        file_saved = os.path.isfile(subm_path)
        self.assertFalse(file_saved)
