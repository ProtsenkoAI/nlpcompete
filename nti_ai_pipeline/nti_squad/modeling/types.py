import torch
from typing import *


UnprocFeatures = Tuple[List[str], List[str]]
ModelPreds = Tuple[torch.Tensor, torch.Tensor]
PredProbs = Collection[float]
PredAnswerCharIdxs = List[Tuple[int, int]]
UnprocLabels = Union[None, Tuple[List[int], List[int]]]
ProcLabelsTokenIdxs = Tuple[torch.Tensor, torch.Tensor]

ProcFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TrueTokenIdxs = Tuple[List[int], List[int]]
ProcLabels = Tuple[torch.Tensor, torch.Tensor]

SubmSamplePredWithProbs = NamedTuple("SubmPredWithProbs", [("probs", float), ("preds", Any)])
SubmBatchPredWithProbs = List[SubmSamplePredWithProbs]
SubmBatchPred = NamedTuple("SubmBatchPred", [("preds", Any)])
