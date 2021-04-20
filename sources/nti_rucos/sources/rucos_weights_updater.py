from pipeline.training import WeightsUpdater
import torch
from torch import nn


class RucosWeightsUpdater(WeightsUpdater):
    def __init__(self, *args, criterion=nn.CrossEntropyLoss(), **kwargs):
        self.criterion = criterion
        super().__init__(*args, **kwargs)

    def get_loss(self, preds: torch.Tensor, labels_after_preproc: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds.view(-1, 2), labels_after_preproc.view(-1).long())
