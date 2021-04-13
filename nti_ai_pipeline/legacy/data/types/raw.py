# Colab python version is 3.6.9, so we need to use this
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import List


class TrainAnswer(TypedDict):
    text: str
    answer_start: int


class TrainQuestion(TypedDict):
    id: str
    question: str
    answers: List[TrainAnswer]


class TrainParagraph(TypedDict):
    context: str
    qas: List[TrainQuestion]


class TrainDataset(TypedDict):
    title: str
    paragraphs: List[TrainParagraph]
