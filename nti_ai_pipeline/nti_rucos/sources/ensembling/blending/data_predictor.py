from torch.utils import data as torch_data
from typing import Any, List

from pipeline.modeling import ModelManager


class DataPredictor:
    # TODO: refactor, add additional attributes to customize functionality (check "probs" thing and think about it)
    ModelPreds = Any

    def predict(self, model_manager: ModelManager, data_loader: torch_data.DataLoader,
                return_raw_model_preds=False) -> List[ModelPreds]:
        preds = []

        for batch in data_loader:
            if return_raw_model_preds:
                batch_preds = model_manager.preproc_forward(batch)
            else:
                # batch_preds_raw = list(model_manager.predict_postproc(batch))
                batch_preds_raw = model_manager.predict_postproc(batch)
                batch_preds = [sample.probs for sample in batch_preds_raw]
            preds += batch_preds
        return preds
