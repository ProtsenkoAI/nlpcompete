from typing import List, Union
from abc import ABC, abstractmethod

from .dataset_types import Sample
from .parsed import RucosParsedParagraph, RucosParsedCandidate


class RucosStandardDataset(ABC):
    def __init__(self, data, switch_texts=True):
        self.switch_texts = switch_texts
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[RucosParsedParagraph]) -> List[Sample]:
        result: List[dict] = []
        for paragraph in data:
            for candidate in paragraph.candidates:
                sample = self.create_sample(paragraph, candidate)
                if sample is not None:
                    result.append(sample)
        return result

    @abstractmethod
    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> Sample:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[Sample, List[Sample]]:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)
