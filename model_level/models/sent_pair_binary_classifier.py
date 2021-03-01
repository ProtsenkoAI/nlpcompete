import transformers
import torch.nn as nn
import torch

from typing import Tuple, List, Union


class SentPairBinaryClassifier(nn.Module):
    # TODO: (architecture) make transformer model to be a composition of transformer and head
    def __init__(self, mname: str, cache_dir="./cache_models/",
                 droprate=0.3, head_nlayers=1, head_nneurons=768):
        """
        :param mname: a Model name listed in https://huggingface.co/models
        """
        super().__init__()
        self.mname = mname
        self.cache_dir = cache_dir
        self.droprate = droprate
        self.head_nlayers = head_nlayers
        self.head_nneurons = head_nneurons

        self.transformer = self._load_transformer()
        self.transformer_out_size = self._get_transformer_out_size(self.transformer)

        self.head = self._create_head(droprate, head_nlayers, head_nneurons)

    def get_init_kwargs(self):
        return {"mname": self.mname,
                "cache_dir": self.cache_dir,
                "droprate": self.droprate,
                "head_nlayers": self.head_nlayers,
                "head_nneurons": self.head_nneurons}

    def _get_transformer_out_size(self, transformer: transformers.PreTrainedModel) -> int:
        """To create head of model we need to know output shape of transformer that's dependent on
        transformer that is being used. If it's"""
        config = transformer.config.to_dict()
        return config["hidden_size"]

    def forward(self, transformer_inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        x = self.transformer(*transformer_inputs)
        x = x["pooler_output"]  # be attentive: last_hidden_state isn't used for classification
        x = self.head(x)
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
            out_hidden_size = 1

        linear = nn.Linear(inp_hidden_size, out_hidden_size)
        dropout = nn.Dropout(droprate)
        comps = [dropout, linear]
        if is_last:
            comps.append(nn.Sigmoid())
        else:
            activation = nn.ReLU()
            comps.append(activation)

        return comps

    def reset_weights(self, device: Union[None, torch.device] = None) -> None:
        self.transformer = self._load_transformer()
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if not device is None:
            self.to(device)

    def _load_transformer(self) -> transformers.PreTrainedModel:
        return transformers.AutoModel.from_pretrained(self.mname, cache_dir=self.cache_dir)

    def get_head(self) -> nn.Module:
        return self.head

    def get_transformer(self) -> transformers.PreTrainedModel:
        return self.transformer
