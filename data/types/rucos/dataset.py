from typing import NamedTuple


class RucosSample(NamedTuple):
    text1: str
    text2: str
    label: int

class RucosSubmissionSample(NamedTuple):
    text1: str
    text2: str
    question_idx: int
    start: int
    end: int
    placeholder: str
