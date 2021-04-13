import transformers
from torch import nn
import torch
from typing import Optional
from abc import ABC, abstractmethod


class ModelWithTransformer(nn.Module, ABC):
    def __init__(self, transformer_name: str, pretrain_path: str, device: Optional[torch.device] = None):
        super().__init__()
        self.mname = transformer_name
        self.pretrain_path = pretrain_path
        self.transformer = self._load_transformer()
        self.device = device
        self.to(device)

    def reset_weights(self) -> None:
        self.transformer = self._load_transformer()
        for layer in self.get_head().children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if self.device is not None:
            self.to(self.device)

    def _load_transformer(self):
        # TODO: rewrite in more robust way
        if self.pretrain_path is None:
            return transformers.AutoModel.from_pretrained(self.mname)
        else:
            bert_state = torch.load(self.pretrain_path)
            bert_lm = transformers.BertForMaskedLM.from_pretrained(self.mname)
            bert_lm.load_state_dict(bert_state)

            bert = transformers.AutoModel.from_pretrained(self.mname)
            bert.bert = bert_lm.bert
            return bert

    def get_transformer_out_size(self, transformer: transformers.PreTrainedModel) -> int:
        """To create head of model we need to know output shape of transformer that's dependent on
        transformer that is being used. If it's"""
        config = transformer.config.to_dict()
        return config["hidden_size"]

    def get_transformer(self) -> transformers.PreTrainedModel:
        return self.transformer

    @abstractmethod
    def get_head(self) -> nn.Module:
        ...
