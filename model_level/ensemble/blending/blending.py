from typing import List, Union, Tuple, Iterable
from pprint import pprint

import torch

from ..unbatching_processor import UnbatchingProcessor


# TODO: import from types when these moved from model_level/managing_model.py to types
UnprocLabels = Union[None, Tuple[List[int], List[int]]]
UnprocFeatures = Tuple[List[str], List[str]]
ModelPreds = Tuple[torch.Tensor, torch.Tensor]
ProcLabelsTokenIdxs = Tuple[torch.Tensor, torch.Tensor]

class BlendingModelManager:
    def __init__(self, weights: List[float], processor: UnbatchingProcessor):
        # TODO: sum of floats can be not accurate
        if sum(weights) != 1:
            raise ValueError('weights sum doesn\'t equal 1')
        self._weights = weights
        self._processor = processor

    def preproc_forward(self, models_preds: List[ModelPreds]) -> Iterable[ModelPreds]:
        if len(models_preds) != len(self._weights):
            raise ValueError('model_preds and self.weights mismatch')
        splitted_preds: Iterable[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = map(lambda pred: self._processor.preprocess(pred), models_preds)
        for i, samples_preds in enumerate(zip(self._weights, splitted_preds)): # type: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
            print(f'bruh {i}')
            pprint(samples_preds)
