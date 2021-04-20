import torch
from typing import *


# UnprocFeatures = Tuple[List[str], List[str]]
UnprocFeatures = NamedTuple("UnprocFeatures", [("text", str), ("question", str)])
ModelPreds = Tuple[torch.Tensor, torch.Tensor]
PredProbs = Collection[float]
PredAnswerCharIdxs = List[Tuple[int, int]]
UnprocLabels = Union[None, Tuple[List[int], List[int]]]
ProcLabelsTokenIdxs = Tuple[torch.Tensor, torch.Tensor]

ProcFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TrueTokenIdxs = Tuple[List[int], List[int]]
ProcLabels = Tuple[torch.Tensor, torch.Tensor]

SubmSamplePredWithProbsAndStartEnds = NamedTuple("SubmPredWithProbs", [("probs", float), ("answer_start", int),
                                                                       ("answer_end", int), ("preds", Any)])
SubmSamplePredWithProbs = NamedTuple("SubmPredWithProbs", [("probs", float), ("preds", Any)])
SubmBatchPredWithProbsAndStartEnds = List[SubmSamplePredWithProbsAndStartEnds]
SubmBatchPred = NamedTuple("SubmBatchPred", [("preds", Any)])
