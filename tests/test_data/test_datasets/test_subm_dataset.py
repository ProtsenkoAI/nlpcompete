import unittest

from ...helpers import config, std_objects

config = config.TestsConfig()


class SharedObjects:
    """The class is used to speed up testing"""
    def __init__(self):
        self.nrows = 20
        self.container = std_objects.get_container(nrows=self.nrows)


shared_objs = SharedObjects()


class TestSubmDataset(unittest.TestCase):
    def test_len(self):
        dataset = std_objects.get_subm_dataset(shared_objs.container)
        self.assertGreaterEqual(len(dataset), shared_objs.nrows)

    def test_getitem(self):
        dataset = std_objects.get_subm_dataset(shared_objs.container)

        first_elem = dataset[0]
        last_elem = dataset[len(dataset) - 1]
        self._check_elem_structure(first_elem)
        self._check_elem_structure(last_elem)

        with self.assertRaises(Exception):
            _ = dataset[len(dataset)]

    def _check_elem_structure(self, elem):
        (quest_id, context, question) = elem
        self.assertIsInstance(quest_id, str)
        self.assertIsInstance(context, str)
        self.assertIsInstance(question, str)
