import torch.nn as nn
import torch
from typing import Tuple, List, Optional

from pipeline.modeling import ModelWithTransformer


class TransformerQA(ModelWithTransformer):
    def __init__(self, mname: str, cache_dir="./cache_models/",
                 droprate=0.3, head_nlayers=1, head_nneurons=768, classif_thresh=0.5,
                 pretrain_path: Optional[str] = None):
        """
        :param mname: a Model name listed in https://huggingface.co/models
        """
        super().__init__(transformer_name=mname, cache_dir=cache_dir, pretrain_path=pretrain_path)
        self.mname = mname
        self.cache_dir = cache_dir
        self.droprate = droprate
        self.head_nlayers = head_nlayers
        self.head_nneurons = head_nneurons
        self.classification_thresh = classif_thresh

        self.head = self._create_head(droprate, head_nlayers, head_nneurons)

    def get_head(self) -> nn.Module:
        return self.head

    def get_init_kwargs(self):
        return {"mname": self.mname,
                "cache_dir": self.cache_dir,
                "droprate": self.droprate,
                "head_nlayers": self.head_nlayers,
                "head_nneurons": self.head_nneurons}

    def forward(self, transformer_inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        :return: tuple with (start token probabilities, end token probabilities)
        """
        x = self.get_transformer()(*transformer_inputs)
        x = x["last_hidden_state"]  # be attentive: last_hidden_state isn't used for classification
        x = self.get_head()(x)
        return x

    def _create_head(self, droprate: float, n_layers: int, head_nneurons: int) -> nn.Module:
        head_layers = []

        for layer_idx in range(n_layers):
            is_first = layer_idx == 0
            is_last = layer_idx == (n_layers - 1)
            layers = self._create_head_layer_components(is_first, is_last, droprate, head_nneurons)
            head_layers += layers

        return nn.Sequential(*head_layers)

    def _create_head_layer_components(self, is_first: bool, is_last: bool, droprate: float, nneurons: int
                                      ) -> List[nn.Module]:
        inp_hidden_size, out_hidden_size = nneurons, nneurons
        if is_first:
            inp_hidden_size = self.transformer_out_size
        if is_last:
            out_hidden_size = 2

        linear = nn.Linear(inp_hidden_size, out_hidden_size)
        dropout = nn.Dropout(droprate)
        comps = [dropout, linear]

        if not is_last:
            activation = nn.ReLU()
            comps.append(activation)

        return comps
