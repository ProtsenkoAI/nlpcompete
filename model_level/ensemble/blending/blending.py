from typing import List, Union, Tuple

import torch
import numpy as np


# TODO: import from types when these moved from model_level/managing_model.py to types
ModelPreds = Tuple[torch.Tensor, torch.Tensor]

class BlendingModelManager:
    def __init__(self, weights: List[float]):
        # TODO: sum of floats can be not accurate
        if sum(weights) != 1:
            raise ValueError('weights sum doesn\'t equal 1')
        self._weights = np.array(weights)

    def preproc_forward(self, models_preds: List[ModelPreds]) -> ModelPreds:
        if len(models_preds) != len(self._weights):
            raise ValueError('model_preds and self.weights mismatch')
        models_preds_squeezed = []
        for model_preds in models_preds:
            models_preds_squeezed.extend(model_preds)
        tmp = torch.stack(models_preds_squeezed).reshape(
            len(models_preds),
            -1,
            models_preds[0][0].shape[0],
            models_preds[0][0].shape[1]
        )
        blended_probs = (tmp * self._weights[:, None, None, None]).sum(axis=0)
        start_preds, end_preds = blended_probs.split(1, dim=0)
        return torch.squeeze(start_preds), torch.squeeze(end_preds)
