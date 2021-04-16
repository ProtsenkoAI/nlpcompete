from .rucos_standard_dataset import RucosStandardDataset
from .parsed_types import RucosParsedParagraph, RucosParsedCandidate
from nti_rucos.sources.data.dataset_types import RucosSubmissionSample
from .dataset_types import Sample


class RucosSubmDataset(RucosStandardDataset):
    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> Sample:
        sample = RucosSubmissionSample(
                                        text1=paragraph.text1,
                                        text2=candidate.text2,
                                        question_idx=paragraph.idx,
                                        start=candidate.start_char,
                                        end=candidate.end_char,
                                        placeholder=candidate.placeholder
                                    )
        return sample
