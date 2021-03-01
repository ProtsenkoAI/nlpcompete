from typing import NamedTuple

class RucosSampleFeatures(NamedTuple):
    text1: str
    text2: str


class RucosSample(NamedTuple):
    features: RucosSampleFeatures
    label: int


class RucosSubmissionSample(NamedTuple):
    text1: str
    text2: str
    question_idx: int
    start: int
    end: int
    placeholder: str
