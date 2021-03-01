# from typing import List, Union
#
# from ..contain.rucos_contain import RucosDataContainer
# from ..types.rucos.parsed import RucosParsedParagraph
# from ..types.rucos.dataset import RucosEvalSample, RucosSampleFeatures
# from .base_sized_dataset import SizedDataset
#
#
# class RucosStandardDataset(SizedDataset):
#     def __init__(self, container: RucosDataContainer):
#         data = container.get_data()
#         self.samples = self._get_samples(data)
#
#     def _get_samples(self, data: List[RucosParsedParagraph]) -> List[RucosEvalSample]:
#         result: List[RucosEvalSample] = []
#         for paragraph in data:
#             for candidate in paragraph.candidates:
#                 result.append(RucosEvalSample(
#                     question_idx=paragraph.idx,
#                     features=RucosSampleFeatures(text1=paragraph.text1, text2=candidate.text2),
#                     label=candidate.label
#                 ))
#         return result
#
#     def __getitem__(self, item: Union[int, slice]) -> Union[RucosEvalSample, List[RucosEvalSample]]:
#         return self.samples[item]
#
#     def __len__(self) -> int:
#         return len(self.samples)
