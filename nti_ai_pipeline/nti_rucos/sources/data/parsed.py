from typing import NamedTuple, List, Optional


class RucosParsedCandidate(NamedTuple):
    text2: str              # query with replaced placeholder
    label: Optional[int]    # is this placeholder right
    start_char: int
    end_char: int
    placeholder: str        # entity to replace placeholder in query


class RucosParsedParagraph(NamedTuple):
    text1: str              # context
    idx: int                # index
    candidates: List[RucosParsedCandidate]
