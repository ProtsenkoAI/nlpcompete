from torch.utils import data as torch_data
from sklearn.model_selection import train_test_split, KFold

from typing import Tuple, Generator, Any
from .datasets.base_sized_dataset import SizedDataset



class DataAssistant:
    def __init__(self, loader_builder):
        self.loader_builder = loader_builder

    def train_test_split(self, dataset: SizedDataset, test_size=0.2):
        dataset_indexes = list(range(len(dataset)))
        train_idxs, test_idxs = train_test_split(dataset_indexes, test_size=test_size)
        train_loader, test_loader = self._split_train_val_create_loaders(dataset, train_idxs, test_idxs)
        return train_loader, test_loader

    def split_folds(self, dataset, nfolds, max_fold=None,
                    ) -> Generator[Tuple[torch_data.DataLoader, torch_data.DataLoader], Any, None]:
        """When using sklearn.KFold, we can't regulate number of samples in test, it equals len(dataset) // nfolds.
        So if you want to gen 3 folds with test_size=0.1, pass nfolds=10 and stop_fold=3."""
        if max_fold is None:
            max_fold = nfolds
        dataset_indexes = list(range(len(dataset)))
        folds_idxs = KFold(nfolds, shuffle=True).split(dataset_indexes)

        def generator():
            fold_cnt = 0
            for train_idxs, test_idxs in folds_idxs:
                if fold_cnt >= max_fold:
                    break
                train_loader, test_loader = self._split_train_val_create_loaders(dataset, train_idxs, test_idxs)
                fold_cnt += 1
                yield train_loader, test_loader

        return generator()

    def get_without_split(self, dataset, has_answers=True):
        return self._create_loader_with_answers(dataset, has_answers=has_answers)

    def _split_train_val_create_loaders(self, dataset, train_idxs, test_idxs
                                        ) -> Tuple[torch_data.DataLoader, torch_data.DataLoader]:
        train_dataset = torch_data.Subset(dataset, train_idxs)
        val_dataset = torch_data.Subset(dataset, test_idxs)
        train_loader = self._create_loader_with_answers(train_dataset)
        test_loader = self._create_loader_with_answers(val_dataset)
        return train_loader, test_loader


    def _create_loader_with_answers(self, dataset, has_answers=True) -> torch_data.DataLoader:
        return self.loader_builder.build(dataset, has_answers=has_answers)