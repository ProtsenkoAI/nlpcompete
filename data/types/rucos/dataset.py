from typing import NamedTuple

class RucosSampleFeatures(NamedTuple):
    text1: str
    text2: str
    placeholder: str


class RucosSample(NamedTuple):
    features: RucosSampleFeatures
    label: int

class RucosEvalSampleFeatures(NamedTuple):
    question_idx: int
    text1: str
    text2: str
    placeholder: str


class RucosSubmissionSample(NamedTuple):
    text1: str
    text2: str
    question_idx: int
    start: int
    end: int
    placeholder: str
