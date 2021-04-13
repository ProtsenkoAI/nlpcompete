import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from typing import List


class RucosRawEntity(TypedDict):
    start: int
    end: int
    text: str

class RucosRawQuery(TypedDict):
    query: str
    answers: List[RucosRawEntity]


class RucosRawPassage(TypedDict):
    text: str
    entities: List[RucosRawEntity]


class RucosRawParagraph(TypedDict):
    source: str
    passage: RucosRawPassage
    qas: List[RucosRawQuery]
    idx: int
