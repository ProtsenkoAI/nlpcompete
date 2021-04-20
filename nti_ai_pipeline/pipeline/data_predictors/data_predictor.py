from torch.utils import data as torch_data
from typing import Any, List
from abc import ABC, abstractmethod

from pipeline.modeling import ModelManager
from ..util import get_tqdm_obj


class DataPredictor(ABC):
    # TODO: refactor, add additional attributes to customize functionality (check "probs" thing and think about it)
    ModelPreds = Any

    def __init__(self, tqdm_mode="notebook"):
        self.tqdm_obj = get_tqdm_obj(mode=tqdm_mode)

    def predict(self, model_manager: ModelManager, data_loader: torch_data.DataLoader) -> List[ModelPreds]:
        preds = []

        for batch in self.tqdm_obj(data_loader):
            batch_preds = self.pred_batch(model_manager, batch)
            preds += batch_preds
        return preds

    @abstractmethod
    def pred_batch(self, model_manager: ModelManager, batch: Any) -> List:
        ...
