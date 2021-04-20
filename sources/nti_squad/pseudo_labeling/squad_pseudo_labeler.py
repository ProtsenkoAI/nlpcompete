from typing import List, Tuple
from torch.utils.data import DataLoader
import numpy as np

from pipeline.pseudo_labeling import BasePseudoLabeler
from .squad_data_predictor import SquadDataPredictor
from ..data.types_dataset import SampleFeaturesWithAnswers, SampleFeatures
from ..modeling.types import SubmSamplePredWithProbsAndStartEnds


class SQuADPseudoLabeler(BasePseudoLabeler):
    StartsEnds = List[Tuple[int, int]]

    def __init__(self, predictor: SquadDataPredictor, num_samples: int):
        self.num_samples = num_samples
        super().__init__(predictor)

    def get_chosen_samples_idxs_and_labels(self, predictions: List[SubmSamplePredWithProbsAndStartEnds]
                                           ) -> Tuple[List[int], StartsEnds]:
        # TODO: refactor
        """Just choose samples with maximum probability, because transformer is being fitted to predict position thus
        we don't need negative samples
        Returns: indexes and None as labels, because in SQuAD labels aren't needed (we have only text, start and end
        of answer those are obtained from features
        """
        probs, answers_start, answers_end, texts = zip(*predictions)
        top_n_samples_idxs = np.array(probs).argsort()[::-1][:self.num_samples]
        filtered_answers_start_end = [(start, end) for idx, (start, end) in enumerate(zip(answers_start, answers_end))
                                      if idx in top_n_samples_idxs]
        return top_n_samples_idxs, filtered_answers_start_end

    def filter_features(self, loader: DataLoader, idxs: List[int]) -> List[SampleFeatures]:
        samples = []
        sample_idx = 0
        for batch in loader:
            for text, question in zip(batch.text, batch.question):
                if sample_idx in idxs:
                    features = SampleFeatures(text=text, question=question)
                    samples.append(features)
                sample_idx += 1
        return samples

    def union_features_with_labels(self, features, labels: StartsEnds):
        labeled_samples = [SampleFeaturesWithAnswers(features=features, answer_start_end=labels) for features, labels in
                           zip(features, labels)]
        return labeled_samples
