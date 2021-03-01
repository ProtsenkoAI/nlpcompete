import json
from typing import Optional, List

from ..types.rucos.parsed import RucosParsedParagraph, RucosParsedCandidate
from ..types.rucos.raw import RucosRawParagraph, RucosRawEntity, RucosRawQuery, RucosRawPassage


class RucosDataContainer:
    def __init__(self, path: str, nrows: Optional[int] = None):
        self.path = path
        self.nrows = nrows

    def get_data(self) -> List[RucosParsedParagraph]:
        result: List[RucosParsedParagraph] = []
        with open(self.path) as f:
            if self.nrows is not None:
                for i, line in zip(range(self.nrows), f):  # type: int, str
                    result.append(self._parse_paragraph(json.loads(line)))
            else:
                for line in f:
                    result.append(self._parse_paragraph(json.loads(line)))
        return result

    def _parse_paragraph(self, p: RucosRawParagraph) -> RucosParsedParagraph:
        return RucosParsedParagraph(
            text1=p['passage']['text'],
            idx=p['idx'],
            candidates=[self._parse_candidate(c, p['qas'][0], p['passage']) for c in p['passage']['entities']]
        )

    def _parse_candidate(self, entity: RucosRawEntity, query: RucosRawQuery, passage: RucosRawPassage) -> RucosParsedCandidate:
        placeholder = passage['text'][entity['start']:entity['end']]
        is_answer = any(ans['start'] == entity['start'] and ans['end'] == entity['end'] for ans in query['answers'])
        return RucosParsedCandidate(
            text2=query['query'].replace(
                '@placeholder',
                placeholder,
                1
            ),
            label=1 if is_answer else 0,
            start_char=entity['start'],
            end_char=entity['end'],
            placeholder=placeholder
        )
