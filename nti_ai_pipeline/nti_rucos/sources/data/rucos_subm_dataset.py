from .rucos_candidates_dataset import RucosCandidatesDataset
from .parsed_types import RucosParsedParagraph, RucosParsedCandidate
from nti_rucos.sources.data.dataset_types import RucosSubmissionSample
from .dataset_types import Sample


class RucosSubmDataset(RucosCandidatesDataset):
    def create_sample(self, paragraph: RucosParsedParagraph, candidate: RucosParsedCandidate) -> Sample:
        text1 = paragraph.text1
        text2 = candidate.text2
        if self.switch_texts:
            text1, text2 = text2, text1
        sample = RucosSubmissionSample(
                                        text1=text1,
                                        text2=text2,
                                        question_idx=paragraph.idx,
                                        start=candidate.start_char,
                                        end=candidate.end_char,
                                        placeholder=candidate.placeholder
                                    )
        return sample
