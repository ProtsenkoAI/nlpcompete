from typing import Sequence, List, Union
import numpy as np


class RegressionResultsBlender:
    r"""
    Takes already made predictions of every model and returns blended predictions.
    """
    def blend(self, models_preds: List[Sequence], models_weights: List[Union[float, int]]) -> np.array:
        self._validate_preds_and_weights(models_preds, models_weights)
        preds_arr = np.array(models_preds)
        normalized_weights = np.array(models_weights) / sum(models_weights)
        preds_weighted = preds_arr * normalized_weights
        return preds_weighted

    def _validate_preds_and_weights(self, preds, weights):
        if len(preds) != len(weights):
            raise ValueError("Number of model weights not equal to number of predictions objects.")
