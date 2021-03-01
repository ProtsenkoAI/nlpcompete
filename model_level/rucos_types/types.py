import torch
from typing import *

UnprocFeatures = Tuple[List[str], List[str]]
UnprocSubmFeatures = Tuple[List[str], List[str], List[int], List[int], List[int], List[str]] # text1, text2, question_idx, start, end, placeholder
UnprocLabels = List[int]
ProcFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ProcLabels = torch.Tensor
ModelPreds = torch.Tensor
ProcSubmPreds = Tuple[List[int], List[float], List[int], List[int], List[str]] # text_id, probability, start, end, placeholder