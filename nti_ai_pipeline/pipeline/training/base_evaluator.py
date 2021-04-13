from abc import ABC, abstractmethod
from pipeline.modeling import ModelManager
from torch.utils.data import DataLoader


class Validator(ABC):
    # TODO
    @abstractmethod
    def eval(self, manager: ModelManager, test_loader: DataLoader) -> float:
        ...
