import unittest
from torch.utils import data as torch_data

from ..helpers import config, std_objects

config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        # self.container = std_objects.get_container(nrows=50)
        self.dataset = std_objects.get_train_dataset(nrows=50)


shared_objs = SharedObjects()


class TestDataAssistant(unittest.TestCase):
    def test_train_test_split(self):
        train_loader, val_loader = std_objects.get_data_assistant().train_test_split(shared_objs.dataset, test_size=0.2)
        self.assertIsInstance(train_loader, torch_data.DataLoader)
        self.assertIsInstance(val_loader, torch_data.DataLoader)

    def test_iterate_3_folds_split(self):
        splitter = std_objects.get_data_assistant()
        niters = 0
        nfolds = 3
        max_fold = 2
        for train_loader, val_loader in splitter.split_folds(shared_objs.dataset, nfolds=nfolds, max_fold=max_fold):
            self.assertIsInstance(train_loader, torch_data.DataLoader)
            self.assertIsInstance(val_loader, torch_data.DataLoader)
            niters += 1
        self.assertEqual(niters, max_fold)

    def test_create_only_train(self):
        splitter = std_objects.get_data_assistant()
        train_loader = splitter.get_without_split(shared_objs.dataset)
        self.assertIsInstance(train_loader, torch_data.DataLoader)
