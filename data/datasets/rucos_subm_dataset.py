from typing import List, Union

from .base_sized_dataset import SizedDataset
from ..contain.rucos_contain import RucosDataContainer
from ..types.rucos.parsed import RucosParsedParagraph
from ..types.rucos.dataset import RucosSubmissionSample


class RucosSubmDataset(SizedDataset):
    def __init__(self, container: RucosDataContainer, switch_texts=False):
        data = container.get_data()
        self.switch_texts = switch_texts
        self._samples = self._get_samples(data)

    def _get_samples(self, data: List[RucosParsedParagraph]) -> List[RucosSubmissionSample]:
        result: List[RucosSubmissionSample] = []
        for paragraph in data:
            for candidate in paragraph.candidates:
                text1 = paragraph.text1
                text2 = candidate.text2
                if self.switch_texts:
                    text2, text1 = text1, text2

                result.append(RucosSubmissionSample(
                    text1=text1,
                    text2=text2,
                    question_idx=paragraph.idx,
                    start=candidate.start_char,
                    end=candidate.end_char,
                    placeholder=candidate.placeholder
                ))
        return result

    def __getitem__(self, item: Union[int, slice]) -> Union[RucosSubmissionSample, List[RucosSubmissionSample]]:
        return self._samples[item]

    def __len__(self) -> int:
        return len(self._samples)
