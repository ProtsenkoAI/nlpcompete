from typing import List

from pipeline.data_predictors import DataPredictor
from pipeline.modeling import ModelManager
from ..data.types_dataset import SampleFeaturesWithId


class SquadDataPredictor(DataPredictor):
    def pred_batch(self, model_manager: ModelManager, batch: SampleFeaturesWithId) -> List:
        batch_preds = model_manager.predict_postproc((batch.text, batch.question),
                                                     postproc_kwargs={"return_probs_and_start_ends": True})
        return batch_preds
