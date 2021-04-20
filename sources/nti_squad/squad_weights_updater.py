from pipeline.training import WeightsUpdater
import torch
from torch import nn


class QAWeightsUpdater(WeightsUpdater):
    def __init__(self, *args, criterion=nn.CrossEntropyLoss(), **kwargs):
        self.criterion = criterion
        super().__init__(*args, **kwargs)

    def get_loss(self, preds: torch.Tensor, labels_after_preproc: torch.Tensor) -> torch.Tensor:
        start_labels, end_labels = labels_after_preproc
        start_probs, end_probs = preds
        loss_start = self.criterion(start_probs, start_labels)
        loss_end = self.criterion(end_probs, end_labels)
        loss = (loss_start + loss_end) / 2
        return loss
