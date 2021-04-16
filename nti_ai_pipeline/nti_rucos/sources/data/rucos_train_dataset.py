from .rucos_standard_dataset import RucosStandardDataset
from .dataset_types import RucosSampleFeatures
from .dataset_types import Sample
from .parsed_types import RucosParsedParagraph, RucosParsedCandidate


class RucosTrainDataset(RucosStandardDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, )

    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> Sample:
        text1 = paragraph.text1
        text2 = candidate.text2
        if self.switch_texts:
            text2, text1 = text1, text2
        return [RucosSampleFeatures(text1=text1, text2=text2,
                                    placeholder=candidate.placeholder),
                candidate.label]
