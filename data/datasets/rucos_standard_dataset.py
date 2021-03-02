from typing import List, Union

from ..contain.rucos_contain import RucosDataContainer
from ..types.rucos.parsed import RucosParsedParagraph
from ..types.rucos.dataset import RucosSample, RucosSampleFeatures, RucosEvalSampleFeatures
from .base_sized_dataset import SizedDataset


class RucosStandardDataset(SizedDataset):
    def __init__(self, container: RucosDataContainer, return_id=False, switch_texts=True):
        data = container.get_data()
        self.return_id = return_id
        self.switch_texts = switch_texts
        self.samples = self._get_samples(data)

    def _get_samples(self, data: List[RucosParsedParagraph]) -> List[dict]:
        result: List[dict] = []
        for paragraph in data:
            for candidate in paragraph.candidates:
                text1 = paragraph.text1
                text2 = candidate.text2
                if self.switch_texts:
                    text2, text1 = text1, text2
                if self.return_id:
                    # NOTE: changed order of texts to place answer at start
                    result.append({"features": RucosEvalSampleFeatures(text1=text1, text2=text2,
                                                                       placeholder=candidate.placeholder,
                                                                       question_idx=paragraph.idx),
                                   "label": candidate.label})
                else:
                    result.append({"question_idx": paragraph.idx,
                                   "features": RucosSampleFeatures(text1=text1, text2=text2,
                                                                    placeholder=candidate.placeholder),
                                   "label": candidate.label})
        return result

    def __getitem__(self, item: Union[int, slice]) -> Union[RucosSample, List[RucosSample]]:
        sample = self.samples[item]
        features = sample["features"]
        label = sample["label"]

        return RucosSample(features, label)

    def __len__(self) -> int:
        return len(self.samples)
