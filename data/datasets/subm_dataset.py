from .base_unlabeled_dataset import UnlabeledDataset
from ..types.dataset import SampleFeaturesWithId


class SubmDataset(UnlabeledDataset):
    def __getitem__(self, idx: int) -> SampleFeaturesWithId:
        features = self.samples[idx]
        return SampleFeaturesWithId(features["id"], features["text"], features["question"])
