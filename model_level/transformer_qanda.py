import transformers
import os.path as path
import torch.nn as nn
import torch


class TransformerQA(nn.Module):
    # TODO: (architecture) make transformer model to be a composition of transformer and head
    def __init__(self, mname, cache_dir="./cache_models/",
                 droprate=0.3, head_nlayers=1, head_nneurons=768):
        super().__init__()
        self.mname = mname
        self.cache_dir = cache_dir
        self.classification_thresh = 0.5

        self.transformer = self._load_transformer()
        self.transformer_out_size = self._get_transformer_out_size(self.transformer)

        self.head = self._create_head(droprate, head_nlayers, head_nneurons)

    def _get_transformer_out_size(self, model):
        config = model.config.to_dict()
        return config["hidden_size"]

    def forward(self, transformer_inputs):
        x = self.transformer(*transformer_inputs)
        x = x["last_hidden_state"]  # be attentive: last_hidden_state isn't used for classification
        x = self.head(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def _create_head(self, droprate, n_layers, head_nneurons):
        head_layers = []
    
        for layer_idx in range(n_layers):
            is_first = layer_idx == 0
            is_last = layer_idx == (n_layers - 1)
            layers = self._create_head_layer_components(is_first, is_last, droprate, head_nneurons)
            head_layers += layers

        return nn.Sequential(*head_layers)

    def _create_head_layer_components(self, is_first, is_last, droprate, nneurons):
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

    def reset_weights(self, device=None):
        self.transformer = self._load_transformer()
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                print("RESETTING", layer)
                layer.reset_parameters()

        if not device is None:
            self.to(device)

    def _load_transformer(self):
        return transformers.AutoModel.from_pretrained(self.mname, cache_dir=self.cache_dir)

    def get_head(self):
        return self.head

    def get_transformer(self):
        return self.transformer
