from .base_unlabeled_dataset import UnlabeledDataset
from ..types.dataset import SampleFeaturesWithSampleIdx


class PseudoLabelingDataset(UnlabeledDataset):
    def __getitem__(self, idx: int) -> SampleFeaturesWithSampleIdx:
        features = self.samples[idx]
        return SampleFeaturesWithSampleIdx(features["text"], features["question"], idx)
