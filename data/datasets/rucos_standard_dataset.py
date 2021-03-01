from typing import List, Union

from ..contain.rucos_contain import RucosDataContainer
from ..types.rucos.parsed import RucosParsedParagraph
from ..types.rucos.dataset import RucosSample, RucosSampleFeatures, RucosEvalSampleFeatures
from .base_sized_dataset import SizedDataset


class RucosStandardDataset(SizedDataset):
    def __init__(self, container: RucosDataContainer, return_id=False):
        data = container.get_data()
        self.return_id = return_id
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[RucosParsedParagraph]) -> List[dict]:
        result: List[dict] = []
        for paragraph in data:
            for candidate in paragraph.candidates:
                if self.return_id:
                    result.append({"features": RucosEvalSampleFeatures(text1=paragraph.text1, text2=candidate.text2,
                                                                       question_idx=paragraph.idx),
                                   "label": candidate.label})
                else:
                    result.append({"question_idx": paragraph.idx,
                                   "features": RucosSampleFeatures(text1=paragraph.text1, text2=candidate.text2),
                                   "label": candidate.label})
        return result

    def __getitem__(self, item: Union[int, slice]) -> Union[RucosSample, List[RucosSample]]:
        sample = self.samples[item]
        features = sample["features"]
        label = sample["label"]

        return RucosSample(features, label)

    def __len__(self) -> int:
        return len(self.samples)
