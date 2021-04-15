import torch.nn as nn
from pipeline.modeling import ModelWithTransformer
import torch

from typing import Tuple


class SentPairBinaryClassifier(ModelWithTransformer):
    # TODO: maybe pass pretrained weights not in file but more simple way (nn.Module obj)
    # TODO: (architecture) make transformer model to be a composition of transformer and head
    def __init__(
            self,
            mname: str,
            droprate: int = 0.3,
            head_nlayers: int = 1,
            head_nneurons: int = 768,
            use_hidden_pooling: bool = False,
            transformer_weights_path=None,
            use_ner=False,
            ner_out_len=7
    ):
        """
        :param mname: a Model name listed in https://huggingface.co/models
        """
        super().__init__(mname, transformer_weights_path)
        self.droprate = droprate
        self.head_nlayers = head_nlayers
        self.head_nneurons = head_nneurons
        self.use_hidden_pooling = use_hidden_pooling

        self.use_ner = use_ner
        self.ner_out_len = ner_out_len

        self.head = self._create_head(droprate, head_nlayers, head_nneurons)

    def get_init_kwargs(self):
        return {"mname": self.mname,
                "cache_dir": self.cache_dir,
                "droprate": self.droprate,
                "head_nlayers": self.head_nlayers,
                "head_nneurons": self.head_nneurons,
                "transformer_weights_path": self.transformer_weights_path,
                "use_ner": self.use_ner,
                "ner_out_len": self.ner_out_len,
                }

    def forward(self, inp) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        if self.use_ner:
            transformer_inputs, ner_probs = inp
        else:
            transformer_inputs = inp
            
        x = self.get_transformer()(*transformer_inputs)
        # print(x)
        if self.use_hidden_pooling:
            x = x['last_hidden_state']
        else:
            x = x["pooler_output"]  # be attentive: last_hidden_state isn't used for classification
        if self.use_ner:
            x = torch.cat([x, ner_probs], dim=1)
            assert len(x) == len(ner_probs)
        x = self.head(x)
        return x

    def _create_head(self, droprate: float, n_layers: int, head_nneurons: int) -> nn.Module:
        # TODO: refactor creating head and maybe push something to base_model (or other base class)
        transformer_out_size = self.get_transformer_out_size(self.transformer)
        layers = []
        if self.use_hidden_pooling:
            layers.append(nn.AdaptiveAvgPool2d(output_size=(1, transformer_out_size)))
        layers.append(nn.Dropout(p=droprate))
        if n_layers > 2:
            raise ValueError(n_layers)
        elif n_layers == 1:
            if self.use_ner:
                layers.append(nn.Linear(transformer_out_size + self.ner_out_len, 2))
            else:
                layers.append(nn.Linear(transformer_out_size, 2))
        elif n_layers == 2:
            if self.use_ner:
                layers.append(nn.Linear(transformer_out_size + self.ner_out_len, head_nneurons))
            else:
                layers.append(nn.Linear(transformer_out_size, head_nneurons))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=droprate))
            layers.append(nn.Linear(head_nneurons, 2))

        return nn.Sequential(*layers)

    def get_head(self) -> nn.Module:
        return self.head
