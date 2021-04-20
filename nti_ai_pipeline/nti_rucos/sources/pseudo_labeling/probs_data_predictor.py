from typing import List

from pipeline.data_predictors import DataPredictor
from pipeline.modeling import ModelManager
from ..modeling.types import ProcSubmPreds


class ProbsDataPredictor(DataPredictor):
    def pred_batch(self, model_manager: ModelManager, batch: ProcSubmPreds) -> List:
        batch_preds = model_manager.predict_postproc(batch)
        probs = [sample.probs for sample in batch_preds]
        return probs
