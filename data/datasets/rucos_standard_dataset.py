from typing import List, Union

from ..contain.rucos_contain import RucosDataContainer
from ..types.rucos.parsed import RucosParsedParagraph
from ..types.rucos.dataset import RucosSample
from .base_sized_dataset import SizedDataset


class RucosStandardDataset(SizedDataset):
    def __init__(self, container: RucosDataContainer):
        data = container.get_data()
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[RucosParsedParagraph]) -> List[RucosSample]:
        result: List[RucosSample] = []
        for paragraph in data:
            for candidate in paragraph.candidates:
                result.append(RucosSample(
                    text1=paragraph.text1,
                    text2=candidate.text2,
                    label=candidate.label
                ))
        return result

    def __getitem__(self, item: Union[int, slice]) -> Union[RucosSample, List[RucosSample]]:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)
