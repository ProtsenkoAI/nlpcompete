from torch.utils import data as torch_data
from typing import List, Sequence
import numpy as np

from pipeline.modeling import ModelManager


class BigModelsBlendManager:
    """
    Supports incrementally adding models to blend and then calculating blend, when all models were added.
    Arranged in this way because usually we have several big models that will not fit in memory at once,
    thus we have to load them sequentially to make predictions.
    """
    BlendRes = np.array

    def __init__(self, data_loader: torch_data.DataLoader, data_predictor, blender):
        self.data_loader = data_loader
        self.data_predictor = data_predictor
        self.blender = blender
        self.predictions: List[Sequence] = []
        self.model_weights: List[float] = []

    def add_model(self, model_manager: ModelManager, model_weight: float):
        self.model_weights.append(model_weight)
        model_preds = self.data_predictor.predict(model_manager, self.data_loader)
        self.predictions.append(model_preds)

    def calc_blend(self) -> BlendRes:
        return self.blender.blend(self.predictions, self.model_weights)
