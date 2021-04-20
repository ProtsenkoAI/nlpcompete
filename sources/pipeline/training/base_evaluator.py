from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Any, Tuple

from pipeline.modeling import ModelManager
from pipeline.util import get_tqdm_obj


class Validator(ABC):
    AllPreds = Any
    AllLabels = Any
    Preds = Any
    Labels = Any

    def __init__(self, tqdm_mode="notebook"):
        self.tqdm = get_tqdm_obj(mode=tqdm_mode)

    def eval(self, manager: ModelManager, test_loader: DataLoader) -> float:
        for batch in self.tqdm(test_loader, desc="eval"):
            batch_preds, labels = self.pred_batch(manager, batch)
            self.store_batch_res(batch_preds, labels)
        test_preds, test_labels = self.get_all_preds_and_labels()
        self.clear_accumulated_preds_and_labels()
        return self.calc_metric(test_preds, test_labels)

    @abstractmethod
    def pred_batch(self, manager: ModelManager, batch) -> Tuple[Preds, Labels]:
        ...

    @abstractmethod
    def store_batch_res(self, preds, labels):
        ...

    @abstractmethod
    def get_all_preds_and_labels(self) -> Tuple[AllPreds, AllLabels]:
        ...

    @abstractmethod
    def calc_metric(self, preds: AllPreds, labels: AllLabels) -> float:
        ...

    @abstractmethod
    def clear_accumulated_preds_and_labels(self):
        ...
