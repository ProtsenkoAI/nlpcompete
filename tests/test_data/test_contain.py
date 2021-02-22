import unittest

from ..helpers import config, std_objects

config = config.TestsConfig()


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.train_path = config.train_path
        self.nonexistent_path = config.train_path + "bebebe_s_bababa"

    def test_data_format(self):
        nrows = 20
        container = std_objects.get_container(path=self.train_path, nrows=nrows)
        data = container.get_data()
        self.assertEqual(len(data), nrows)
        for text, text_data in data:
            self.assertIsInstance(text, str)
            for question, question_data in text_data:
                self.assertIsInstance(question, str)
                self.assertIsInstance(question_data["id"], str)
                for start_idx, end_idx in question_data["answers"]:
                    self.assertIsInstance(start_idx, int)
                    self.assertIsInstance(end_idx, int)

    def test_raises_get_nonexistent_file_data(self):
        container = std_objects.get_container(path=self.nonexistent_path)
        with self.assertRaises(Exception):
            container.get_data()
