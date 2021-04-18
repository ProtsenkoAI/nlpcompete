from .rucos_candidates_dataset import RucosCandidatesDataset
from .dataset_types import RucosSampleFeatures
from .dataset_types import RucosSample
from .parsed_types import RucosParsedParagraph, RucosParsedCandidate


class RucosTrainDataset(RucosCandidatesDataset):
    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> RucosSample:
        text1 = paragraph.text1
        text2 = candidate.text2
        if self.switch_texts:
            text1, text2 = text2, text1
        return RucosSample(RucosSampleFeatures(text1=text1, text2=text2,
                                               placeholder=candidate.placeholder),
                           candidate.label)
