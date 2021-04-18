from .rucos_candidates_dataset import RucosCandidatesDataset
from .dataset_types import RucosEvalSampleFeatures
from .dataset_types import RucosSample
from .parsed_types import RucosParsedParagraph, RucosParsedCandidate


class RucosValDataset(RucosCandidatesDataset):
    def __init__(self, data, switch_texts=True):
        super().__init__(data, switch_texts=switch_texts)

    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> RucosSample:
        text1 = paragraph.text1
        text2 = candidate.text2
        if self.switch_texts:
            text1, text2 = text2, text1

        sample = RucosSample(RucosEvalSampleFeatures(text1=text1, text2=text2,
                                                     placeholder=candidate.placeholder,
                                                     question_idx=paragraph.idx),
                             candidate.label)
        return sample
