from typing import Union, Tuple
import torch
from torch import nn


class MockModel(nn.Module):
    r"""
    Model imitating model for NTI RUCOS to test other components without problems with RAM
    """
    def __init__(self, *args, use_ner=False, **kwargs):
        super().__init__()
        self.use_ner = use_ner
        self.device = torch.device("cpu")
        self.transformer = nn.Linear(512, 2)
        self.head = nn.Linear(2, 2)

    def reset_weights(self):
        ...

    def forward(self, features: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if self.use_ner:
            ner_out, transformer_inp = features
        else:
            transformer_inp = features
        first_tensor_from_features = transformer_inp[0].float()
        transformer_out = self.get_transformer()(first_tensor_from_features)
        return self.get_head()(transformer_out)

    def to(self, device):
        self.device = device
        self.get_head().to(device)
        super().to(device)

    def get_transformer(self):
        return self.transformer

    def get_head(self):
        return self.head

    def get_init_kwargs(self):
        return {}