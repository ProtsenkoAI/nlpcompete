# from typing import List
#
# from .base_sized_dataset import SizedDataset
#
# from ..contain import DataContainer
# from ..types.parsed import ParsedParagraph
# from ..types.dataset import SampleWithId
# from ..types.dataset import SampleFeatures
#
#
# class UnlabeledDataset(SizedDataset):
#     def __init__(self, container: DataContainer):
#         data = container.get_data()
#         self.samples = self._get_samples(data)
#
#     def _get_samples(self, data: List[ParsedParagraph]) -> List[SampleWithId]:
#         samples = []
#         for text, questions in data:
#             for question, question_data in questions:
#                 quest_id = question_data["id"]
#                 sample = SampleWithId(id=quest_id, text=text, question=question)
#                 samples.append(sample)
#
#         return samples
#
#     def __getitem__(self, idx: int) -> SampleFeatures:
#         raise NotImplementedError
#
#     def __len__(self):
#         return len(self.samples)
