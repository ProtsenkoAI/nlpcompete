from typing import Union, Tuple
import torch
from torch import nn


class MockModel(nn.Module):
    r"""
    Model imitating model for NTI RUCOS to test other components without problems with RAM
    """
    def __init__(self, *args, use_ner=False, **kwargs):
        self.use_ner = use_ner
        self.device = torch.device("cpu")
        super().__init__()

    def reset_weights(self):
        ...

    def forward(self, features: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if self.use_ner:
            ner_out, transformer_inp = features
        else:
            transformer_inp = features
        batch_size = transformer_inp.shape[0]
        return torch.zeros((batch_size, 2), device=self.device)

    def to(self, device):
        self.device = device
        super().to(device)
