from typing import Any, List

from .data_predictor import DataPredictor
from ..modeling.managing_model import ModelManager


class PostprocDataPredictor(DataPredictor):
    def pred_batch(self, model_manager: ModelManager, batch: Any) -> List:
        batch_preds = model_manager.predict_postproc(batch)
        return batch_preds
