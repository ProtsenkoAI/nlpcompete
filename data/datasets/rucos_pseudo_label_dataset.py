from typing import List, Any, Union

import pandas as pd

from .base_sized_dataset import SizedDataset
from ..contain.rucos_contain import RucosDataContainer
from ..types.rucos.dataset import RucosSample, RucosSampleFeatures


class RucosPseudoLabelDataset(SizedDataset):
    def __init__(self, test_path: str, df: pd.DataFrame, switch_texts: bool = True):
        """
        :param test_path:
        :param df: Accepts DataFrame with columns
            [idx: int, probs: float, start: int, end: int, placeholder: str, label: int]
        :param switch_texts:
        """
        self.switch_texts = switch_texts
        self.container_data = RucosDataContainer(
            test_path,
            has_labels=False,
            text_idxs=df['idx'],
            query_placeholder_union_mode='concatenate'
        ).get_data()
        self.samples = self._get_samples(df)

    def _get_samples(self, df: pd.DataFrame) -> List[RucosSample]:
        result: List[RucosSample] = []
        for i, row in df.iterrows(): # type: int, Any
            current_paragraph = self.container_data[i]
            text1 = current_paragraph.text1
            text2 = next(iter(filter(
                lambda item: item.start_char == row['start'] and item.end_char == row['end'],
                current_paragraph.candidates
            )))
            if self.switch_texts:
                text1, text2 = text2, text1
            result.append(
                RucosSample(
                    features=RucosSampleFeatures(text1, text2, row['placeholder']),
                    label=row['label']
                )
            )
        return result

    def __getitem__(self, item: Union[int, slice]) -> Union[List[RucosSample], RucosSample]:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)
