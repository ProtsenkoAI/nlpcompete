from typing import List, Any, Tuple
from torch.utils.data import DataLoader
import numpy as np

from pipeline.pseudo_labeling import BasePseudoLabeler
from .data.types_dataset import SampleFeatures


class SQuADPseudoLabeler(BasePseudoLabeler):
    def __init__(self, predictor, num_samples: int):
        self.num_samples = num_samples
        super().__init__(predictor)

    def get_chosen_samples_idxs_and_labels(self, predictions: List[float]) -> Tuple[List[int], List[Any]]:
        """Just choose samples with maximum probability, because transformer is being fitted to predict position thus
        we don't need negative samples"""
        top_n_samples_idxs = np.array(predictions).argsort()[::-1][:self.num_samples]
        sample_idxs


    def filter_features(self, loader: DataLoader, idxs: List[int]) -> List[SampleFeatures]:
        ...

    def union_features_with_labels(self, features, labels):
        ...
