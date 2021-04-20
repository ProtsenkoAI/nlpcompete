import transformers
from torch import nn
import torch
from typing import Optional
from abc import ABC, abstractmethod


class ModelWithTransformer(nn.Module, ABC):
    # TODO: maybe add device to subtypes
    def __init__(self, transformer_name: str, pretrain_path: Optional[str] = None,
                 device: Optional[torch.device] = None, cache_dir: Optional[str] = None):
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir
        self.mname = transformer_name
        self.pretrain_path = pretrain_path
        self.transformer = self._load_transformer()
        self.transf_out_size = self._extract_transf_out_size(self.transformer)
        self.to(device)

    def reset_weights(self) -> None:
        self.transformer = self._load_transformer()
        for layer in self.get_head().children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if self.device is not None:
            self.to(self.device)

    def _load_transformer(self) -> transformers.PreTrainedModel:
        # TODO: test that works properly
        if self.pretrain_path is None:
            return transformers.AutoModel.from_pretrained(self.mname, cache_dir=self.cache_dir)
        else:
            bert_state = torch.load(self.transformer_weights_path)

            bert_lm = transformers.BertForMaskedLM.from_pretrained(self.mname)
            bert_lm.load_state_dict(bert_state)

            bert = transformers.AutoModel.from_pretrained(self.mname)
            bert.bert = bert_lm.bert
            return bert

    def _extract_transf_out_size(self, transformer: transformers.PreTrainedModel) -> int:
        config = transformer.config.to_dict()
        return config["hidden_size"]

    def get_transformer_out_size(self):
        """To create head of model we need to know output shape of transformer that's dependent on
        transformer that is being used. If it's"""
        return self.transf_out_size

    def get_transformer(self) -> transformers.PreTrainedModel:
        return self.transformer

    @abstractmethod
    def get_head(self) -> nn.Module:
        ...

    @abstractmethod
    def forward(self, features) -> torch.Tensor:
        ...

    @abstractmethod
    def get_init_kwargs(self) -> dict:
        ...
