from typing import Sequence, List, Union
import numpy as np


class RegressionResultsBlender:
    # TODO
    r"""
    Takes already made predictions of every model and returns blended predictions.
    """
    def __init__(self):
        ...

    def blend(self, models_preds: List[Sequence], models_weights: List[Union[float, int]]) -> np.array:
        raise NotImplementedError
