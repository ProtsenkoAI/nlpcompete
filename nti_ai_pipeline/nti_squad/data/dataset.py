from torch.utils import data as torch_data
from typing import List, Union, Dict

from pipeline.data.base_container import BaseContainer
from .types_parsed import ParsedParagraph
from .types_dataset import SampleWithAnswers, SampleFeatures, SampleFeaturesWithAnswers


class SQuADDataset(torch_data.Dataset):
    def __init__(self, container: BaseContainer):
        data = container.get_data()
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[ParsedParagraph]) -> List[SampleWithAnswers]:
        samples: List[SampleWithAnswers] = []
        for text, questions in data:
            for question, question_data in questions:
                for answer_idxs in question_data["answers"]:
                    start, end = answer_idxs
                    sample = SampleWithAnswers(text=text, question=question,
                                               answer_start=start, answer_end=end)
                    samples.append(sample)

        return samples

    def add_samples(self, new_samples: List[Dict[str, Union[str, int]]]):
        for sample in new_samples:
            sample = SampleWithAnswers(text=sample["text"], question=sample["question"],
                                       answer_start=sample["answer_start"], answer_end=sample["answer_end"])
            self.samples.append(sample)

    def __getitem__(self, idx: int) -> SampleFeaturesWithAnswers:
        sample = self.samples[idx]
        features = SampleFeatures(text=sample["text"], question=sample["question"])
        answer_start_end = sample["answer_start"], sample["answer_end"]
        return SampleFeaturesWithAnswers(features, answer_start_end)

    def __len__(self) -> int:
        return len(self.samples)
