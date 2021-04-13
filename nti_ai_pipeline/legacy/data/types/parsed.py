# Colab python version is 3.6.9, so we need to use this
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import List, Tuple


ParsedAnswer = Tuple[int, int]


class ParsedQuestionAnswers(TypedDict):
    id: str
    answers: List[ParsedAnswer]


ParsedQuestion = Tuple[str, ParsedQuestionAnswers]
ParsedParagraph = Tuple[str, List[ParsedQuestion]]
