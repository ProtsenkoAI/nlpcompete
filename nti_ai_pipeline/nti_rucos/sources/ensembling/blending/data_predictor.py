from torch.utils import data as torch_data
from typing import Any, List

from pipeline.modeling import ModelManager


class DataPredictor:
    ModelPreds = Any

    def predict(self, model_manager: ModelManager, data_loader: torch_data.DataLoader,
                return_raw_model_preds=False) -> List[ModelPreds]:
        preds = []

        for batch in data_loader:
            if return_raw_model_preds:
                batch_preds = model_manager.preproc_forward(batch)
            else:
                batch_preds = model_manager.predict_postproc(batch)
            preds.append(batch_preds)
        return preds
