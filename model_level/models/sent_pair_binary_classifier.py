import transformers
import torch.nn as nn
import torch

from typing import Tuple, List, Union


class SentPairBinaryClassifier(nn.Module):
    # TODO: (architecture) make transformer model to be a composition of transformer and head
    def __init__(
            self,
            mname: str,
            cache_dir: str = "./cache_models/",
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
        super().__init__()
        self.mname = mname
        self.cache_dir = cache_dir
        self.droprate = droprate
        self.head_nlayers = head_nlayers
        self.head_nneurons = head_nneurons
        self.use_hidden_pooling = use_hidden_pooling
        self.transformer_weights_path = transformer_weights_path
        self.transformer = self._load_transformer()

        self.use_ner = use_ner
        self.ner_out_len = ner_out_len

        self.transformer_out_size = self._get_transformer_out_size(self.transformer)

        self.head = self._create_head(droprate, head_nlayers, head_nneurons)

    def get_init_kwargs(self):
        return {"mname": self.mname,
                "cache_dir": self.cache_dir,
                "droprate": self.droprate,
                "head_nlayers": self.head_nlayers,
                "head_nneurons": self.head_nneurons,
                "transformer_weights_path": self.transformer_weights_path}

    def _get_transformer_out_size(self, transformer: transformers.PreTrainedModel) -> int:
        """To create head of model we need to know output shape of transformer that's dependent on
        transformer that is being used. If it's"""
        config = transformer.config.to_dict()
        return config["hidden_size"]

    def forward(self, inp) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        transformer_inputs, ner_probs = inp
        x = self.transformer(*transformer_inputs)
        # print(x)
        if self.use_hidden_pooling:
            x = x['last_hidden_state']
        else:
            x = x["pooler_output"]  # be attentive: last_hidden_state isn't used for classification
        if self.use_ner:
            # print("x", x.shape, x)
            # print("ner_probs", ner_probs.shape, ner_probs)
            x = torch.cat([x, ner_probs], dim=1)
            assert len(x) == len(ner_probs)
        # print("x cat shape", x.shape)
        x = self.head(x)
        return x

    def _create_head(self, droprate: float, n_layers: int, head_nneurons: int) -> nn.Module:
        layers = []
        if self.use_hidden_pooling:
            layers.append(nn.AdaptiveAvgPool2d(output_size=(1, self.transformer_out_size)))
        layers.append(nn.Dropout(p=droprate))
        if n_layers > 2:
            raise ValueError(n_layers)
        elif n_layers == 1:
            if self.use_ner:
                layers.append(nn.Linear(self.transformer_out_size + self.ner_out_len, 2))
            else:
                layers.append(nn.Linear(self.transformer_out_size, 2))
        elif n_layers == 2:
            if self.use_ner:
                layers.append(nn.Linear(self.transformer_out_size + self.ner_out_len, head_nneurons))
            else:
                layers.append(nn.Linear(self.transformer_out_size, head_nneurons))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=droprate))
            layers.append(nn.Linear(head_nneurons, 2))

        return nn.Sequential(*layers)

    def reset_weights(self, device: Union[None, torch.device] = None) -> None:
        self.transformer = self._load_transformer()
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if not device is None:
            self.to(device)

    def _load_transformer(self):
        if self.transformer_weights_path is None:
            return transformers.AutoModel.from_pretrained(self.mname, cache_dir=self.cache_dir)
        else:
            bert_state = torch.load(self.transformer_weights_path)
            bert_lm = transformers.BertForMaskedLM.from_pretrained(self.mname)
            bert_lm.load_state_dict(bert_state)

            bert = transformers.AutoModel.from_pretrained(self.mname)
            bert.bert = bert_lm.bert
            return bert

    def get_head(self) -> nn.Module:
        return self.head

    def get_transformer(self) -> transformers.PreTrainedModel:
        return self.transformer
