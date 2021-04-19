from typing import Optional, Iterable, List, Tuple, Union
from .types import Samples, TrainSubset, ValSubset
from abc import abstractmethod
from sklearn.model_selection import train_test_split, KFold


class BaseContainer:
    def __init__(self, random_state=1):
        self.data = self.collect_data()
        self.random_state = random_state

    @abstractmethod
    def collect_data(self) -> Samples:
        ...

    def get_data(self, start_row: int = 0, nrows: Optional[int] = None,
                 subset_idxs: Optional[Iterable[int]] = None) -> Samples:
        if nrows is not None:
            end_row = start_row + nrows
            return self.data[start_row:end_row]

        elif subset_idxs is not None:
            subset = [elem for idx, elem in enumerate(self.data) if idx in subset_idxs and idx >= start_row]
            return subset

        return self.data[start_row:]

    def kfold(self, nfolds=5, used_folds: Optional[int] = None, shuffle=False) -> List[Tuple[TrainSubset, ValSubset]]:
        """
        :param nfolds: number of folds to split data to. Every fold will contain (nfolds - 1) / nfolds * len(container)
        train elements and len(container) / nfolds val elements.
        :param used_folds: number of folds to return (can, for example, split data to 5 folds and use only 3 of them)
        :param shuffle: whether to shuffle the data in folds or not
        """
        folder = KFold(n_splits=nfolds, random_state=self.random_state, shuffle=shuffle)
        data = self.get_data()
        folds = folder.split(data)
        if not used_folds is None:
            folds = list(folds)[:used_folds]
        return folds

    def train_test_split(self, test_size: Union[int, float] = 0.2, shuffle=False) -> Tuple[TrainSubset, ValSubset]:
        """
        :param: test_size can be proportion of the data (from 0.0 to 1.0), to or number of samples in test(from - to
        len(container)
        :shuffle: whether to shuffle the data or not
        """
        train, test = train_test_split(self.get_data(),
                                       shuffle=shuffle, test_size=test_size, random_state=self.random_state)
        return train, test

    def subset(self, idxs: Iterable[int]):
        data = self.get_data()
        subset = [sample for idx, sample in enumerate(data) if idx in idxs]
        return subset

    def __len__(self):
        return self.data
