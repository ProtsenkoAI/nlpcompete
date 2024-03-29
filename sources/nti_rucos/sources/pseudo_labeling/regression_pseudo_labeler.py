from typing import List, Any, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np

from pipeline.pseudo_labeling import BasePseudoLabeler
from ..data.dataset_types import RucosSampleFeatures, RucosSample


class RegressionPseudoLabeler(BasePseudoLabeler):
    def __init__(self, predictor, chosen_proportion: Optional[float] = 0.2, pos_to_neg_proportion: Optional[float] = 1,
                 neg_thresh: Optional[float] = None, pos_thresh: Optional[float] = None):
        self._validate_betw_0_1(chosen_proportion)
        self._validate_betw_0_1(pos_to_neg_proportion)

        only_one_of_proportion_and_ratio_provided = (chosen_proportion is not None and pos_to_neg_proportion is None or
                                                     chosen_proportion is None and pos_to_neg_proportion is not None)
        if only_one_of_proportion_and_ratio_provided:
            raise ValueError("Should provide both chosen proportion and pos to neg proportion or non of them")

        self.chosen_proportion = chosen_proportion
        self.pos_to_neg_ratio = pos_to_neg_proportion
        self.neg_thresh = neg_thresh
        self.pos_thresh = pos_thresh
        super().__init__(predictor)

    def _validate_betw_0_1(self, proportion):
        if proportion is not None:
            if proportion > 1 or proportion < 0:
                raise ValueError("Proportion should be between 0 and 1")

    def get_chosen_samples_idxs_and_labels(self, predictions: List[float]) -> Tuple[List[int], List[Any]]:
        # TODO: refactor
        if self.chosen_proportion is not None:
            number_of_preds_to_choose = len(predictions) * self.chosen_proportion
            num_pos_samples = int(number_of_preds_to_choose * self.pos_to_neg_ratio)
            num_neg_samples = int(number_of_preds_to_choose / self.pos_to_neg_ratio)
        else:
            num_pos_samples = len(predictions)
            num_neg_samples = len(predictions)

        sorted_preds = np.array(sorted(predictions))

        pos_samples = sorted_preds[-num_pos_samples:]
        neg_samples = sorted_preds[:num_neg_samples]

        if self.neg_thresh is not None:
            neg_samples = neg_samples[neg_samples <= self.neg_thresh]
        if self.pos_thresh is not None:
            pos_samples = pos_samples[pos_samples >= self.pos_thresh]

        neg_idxs = [idx for idx, value in enumerate(predictions) if value in neg_samples]
        pos_idxs = [idx for idx, value in enumerate(predictions) if value in pos_samples]
        pos_and_neg_idxs_union = list(set(neg_idxs + pos_idxs))
        labels = [idx in pos_idxs for idx in pos_and_neg_idxs_union]
        return pos_and_neg_idxs_union, labels

    def filter_features(self, loader: DataLoader, idxs: List[int]) -> List[BasePseudoLabeler.SampleFeatures]:
        chosen_features = []
        sample_idx = 0
        for batch in loader:
            for (text1, text2, placeholder) in zip(batch.text1, batch.text2, batch.placeholder):
                if sample_idx in idxs:
                    chosen_features.append(RucosSampleFeatures(text1, text2, placeholder))
                sample_idx += 1
        return chosen_features

    def union_features_with_labels(self, features, labels):
        out = []
        for sample_features, label in zip(features, labels):
            out.append(RucosSample(sample_features, label))
        return out
