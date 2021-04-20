import torch
from typing import *

UnprocFeatures = Tuple[List[str], List[str], List[str]]  # text1, text2, question_idx, start, end, placeholder
UnprocSubmFeatures = List[Tuple[str, str, int, int, str]]
UnprocLabels = List[int]
ProcFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ProcLabels = torch.Tensor
ModelPreds = torch.Tensor
SubmPred = NamedTuple("SubmPred", [("text_id", int), ("probs", float), ("start", int),
                                   ("end", int), ("placeholder", str)])
ProcSubmPreds = List[SubmPred]
