import transformers
import re
import numpy as np
import torch
from deeppavlov import configs, build_model
from typing import Union, List

from .types import UnprocLabels, UnprocFeatures, ModelPreds, UnprocSubmFeatures, ProcSubmPreds, SubmPred
from pipeline.modeling.types import BatchWithLabels, BatchWithoutLabels
from pipeline.modeling import BaseProcessor


class RucosProcessor(BaseProcessor):
    # TODO: cleaning and refactoring
    def __init__(self, mname: str, use_ner=False, query_goes_first=True):
        """
        :param query_goes_first: indicates whether query (part with missed token)
            goes first or not, needed for tokenizing
        """
        self.mname = mname
        self.maxlen = 512
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.use_ner = use_ner
        self.query_goes_first = query_goes_first
        if self.use_ner:
            self.ner_model = build_model(configs.ner.ner_rus_bert_probas, download=True)

    def get_init_kwargs(self) -> dict:
        return {"mname": self.mname,
                "use_ner": self.use_ner}

    def after_forward(self, raw_model_out):
        return raw_model_out.squeeze(-1)

    def preprocess(self, features: UnprocFeatures, labels: UnprocLabels = None, device: torch.device = None
                   ) -> Union[BatchWithLabels, BatchWithoutLabels]:
        tokenized = self.tokenize(*features)
        tokenized_proc = self._preproc_tokenized(tokenized, device)
        features = tokenized_proc
        if self.use_ner:
            ner_out = self.do_ner(*features)
            ner_proc = torch.tensor(ner_out, dtype=torch.float32)
            ner_proc = ner_proc.to(device, copy=True)
            features = tokenized_proc, ner_proc

        if labels is not None:
            labels_proc = self._create_tensors([labels], device)[0].float()
            return BatchWithLabels(features, labels_proc)
        return BatchWithoutLabels(features)

    def postprocess(self, preds: ModelPreds, src_features: UnprocSubmFeatures) -> ProcSubmPreds:
        text1, text2, text_id, start, end, placeholder = src_features
        probs = preds.squeeze().cpu().detach().numpy()
        out = []
        for prob in probs:
            out.append(SubmPred(text_id, prob, start, end, placeholder))
        return out

    def tokenize(self, *features):
        text1, text2, placeholders = features
        if self.query_goes_first:
            texts = text2
            queries = text1
        else:
            texts = text1
            queries = text2
        encoded = self._tokenize_with_adjust(texts, queries, placeholders)
        return encoded

    def do_ner(self, *features):
        _, _, placeholders = features
        ner_out_raw = self.ner_model(placeholders)[1]
        ner_out_mean = ner_out_raw.mean(axis=1)
        ner_out_max = ner_out_raw.max(axis=1)
        ner_out_first = ner_out_raw[:, 0, :].squeeze()
        ner_out = np.concatenate([ner_out_mean, ner_out_max, ner_out_first], axis=1)
        return ner_out

    def _preproc_tokenized(self, tokenized, device):
        needed_parts = [tokenized["input_ids"],
                        tokenized["attention_mask"],
                        tokenized["token_type_ids"]
                        ]
        tensors = self._create_tensors(needed_parts, device)
        assert len(tensors) == 3
        return tuple(tensors)

    def _create_tensors(self, arrays, device) -> List[torch.Tensor]:
        tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
            if not device is None:
                arr = arr.to(device, copy=True)
            tensors.append(arr)
        return tensors

    def _tokenize_with_adjust(self, texts, queries, placeholders):
        assert (len(texts) == len(placeholders) and len(placeholders) == len(queries))
        adjusted_texts = []
        for placeholder, text, query in zip(placeholders, texts, queries):
            starts_and_ends = [(m.start(), m.end()) for m in re.finditer(placeholder, text)]
            mean_idx = len(starts_and_ends) // 2
            placeholder_start, placeholder_end = starts_and_ends[mean_idx]
            tokenized_query = self._do_tokenize(query)
            ntokens_in_query = np.sum(tokenized_query["attention_mask"])
            tokens_left = self.maxlen - ntokens_in_query

            tokenized_text = self._do_tokenize(text)
            mapping = tokenized_text["offset_mapping"]
            n_tokens_in_text = np.sum(tokenized_query["attention_mask"])
            if n_tokens_in_text <= tokens_left:
                adjusted_texts.append(text)
                continue

            token_starts, token_ends = mapping[0, :n_tokens_in_text, 0], mapping[0, :n_tokens_in_text, 1]
            start_of_placeholder = min(token_starts, key=lambda x: abs(x - placeholder_start))
            end_of_placeholder = min(token_ends, key=lambda x: abs(x - placeholder_end))

            start_adjust_text_token = start_of_placeholder - tokens_left // 2
            end_adjust_text_token = end_of_placeholder + tokens_left // 2
            if end_adjust_text_token >= self.maxlen - 1:
                start_adjust_text_token -= end_adjust_text_token - self.maxlen - 1
                end_adjust_text_token = self.maxlen - 1
            if start_adjust_text_token < 0:
                end_adjust_text_token -= start_adjust_text_token
                start_adjust_text_token = 0

            char_of_start = mapping[0][start_adjust_text_token][0]

            if tokenized_text["attention_mask"][0][end_adjust_text_token] == 0:  # end is out of text
                char_of_end = len(text)
            else:
                char_of_end = mapping[0][end_adjust_text_token][1]

            adjusted_text = text[char_of_start: char_of_end]
            adjusted_texts.append(adjusted_text)
        tokenized = self._do_tokenize(queries, adjusted_texts)
        return tokenized

    def _do_tokenize(self, *texts):
        return self.tokenizer(*texts,
                              max_length=self.maxlen,
                              padding="max_length",
                              truncation=True,
                              return_tensors="np",
                              return_offsets_mapping=True)
