from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from torch.utils.data import DataLoader

from ..data_predictors.data_predictor import DataPredictor  # TODO: move predictor to another place


class BasePseudoLabeler(ABC):
    # TODO: move to pipeline
    SampleFeatures = Any

    def __init__(self, predictor: DataPredictor):
        self.predictor = predictor

    def run(self, model_manager, data_loader):
        preds = self.predictor.predict(model_manager, data_loader)

        chosen_sample_idxs, generated_labels = self.get_chosen_samples_idxs_and_labels(preds)
        chosen_features = self.filter_features(data_loader, chosen_sample_idxs)

        assert len(chosen_features) == len(generated_labels)
        # features_with_labels = list(zip(chosen_features, generated_labels))
        features_with_labels = self.union_features_with_labels(chosen_features, generated_labels)
        return features_with_labels

    @abstractmethod
    def get_chosen_samples_idxs_and_labels(self, predictions: List[Any]) -> Tuple[List[int], List[Any]]:
        ...

    @abstractmethod
    def filter_features(self, loader: DataLoader, idxs: List[int]) -> List[SampleFeatures]:
        ...

    @abstractmethod
    def union_features_with_labels(self, features, labels):
        ...
