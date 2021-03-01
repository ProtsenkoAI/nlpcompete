from typing import List

from .base_sized_dataset import SizedDataset

from ..contain.qa_contain import QADataContainer
from data.types.qa.parsed import ParsedParagraph
from data.types.qa.dataset import SampleWithId, SampleFeaturesWithId


class SubmDataset(SizedDataset):
    def __init__(self, container: QADataContainer):
        data = container.get_data()
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[ParsedParagraph]) -> List[SampleWithId]:
        samples = []
        for text, questions in data:
            for question, question_data in questions:
                quest_id = question_data["id"]
                sample = SampleWithId(id=quest_id, text=text, question=question)
                samples.append(sample)

        return samples

    def __getitem__(self, idx: int) -> SampleFeaturesWithId:
        features = self.samples[idx]
        return SampleFeaturesWithId(features["id"], features["text"], features["question"])

    def __len__(self):
        return len(self.samples)
