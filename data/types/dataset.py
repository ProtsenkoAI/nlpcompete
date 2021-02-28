# Colab python version is 3.6.9, so we need to use this
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import NamedTuple, Tuple


class Sample(TypedDict):
    text: str
    question: str

class SampleWithAnswers(Sample):
    answer_start: int
    answer_end: int

class SampleWithId(Sample):
    id: str

class SampleFeatures(NamedTuple):
    text: str
    question: str

class SampleFeaturesWithAnswers(NamedTuple):
    features: SampleFeatures
    answer_start_end: Tuple[int, int]

class SampleFeaturesWithId(NamedTuple):
    id: str
    text: str
    question: str

class SampleFeaturesWithSampleIdx(NamedTuple):
    text: str
    question: str
    id: int

