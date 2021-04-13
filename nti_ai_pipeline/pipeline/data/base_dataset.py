from torch.utils import data as torch_data
from abc import abstractmethod
from .types import Samples


class BaseDataset(torch_data.Dataset):
    def __init__(self, samples: Samples):
        self._len = len(samples)

    @abstractmethod
    def __getitem__(self, idx: int):
        ...

    def __len__(self):
        return self._len
