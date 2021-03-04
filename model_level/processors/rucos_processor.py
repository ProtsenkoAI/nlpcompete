import transformers
import re
import numpy as np
from deeppavlov import configs, build_model

from ..rucos_types import *

class RucosProcessor:
    # TODO: the class is too large, maybe add assistant components
    def __init__(self, mname: str):
        self.mname = mname
        self.maxlen = 512
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.ner_model = build_model(configs.ner.ner_rus_bert_probas, download=True)

    def get_init_kwargs(self) -> dict:
        return {"mname": self.mname}

    def preprocess(self, features: UnprocFeatures, labels: UnprocLabels=None, device: torch.device=None
                   ) -> Union[ProcFeatures, Tuple[ProcFeatures, ProcLabels]]:
        tokenized, ner_out = self.tokenize_and_do_ner(*features)
        features_proc = self._preproc_tokenized(tokenized, device)
        ner_proc = torch.tensor(ner_out, dtype=torch.float32)
        ner_proc = ner_proc.to(device, copy=True)
        if not labels is None:
            labels_proc = self._create_tensors([labels], device)[0].float()
            return (features_proc, ner_proc), labels_proc
        return (features_proc, ner_proc)

    def postprocess(self, preds: ModelPreds, src_features: UnprocSubmFeatures) -> ProcSubmPreds:
        text1, text2, text_id, start, end, placeholder = src_features
        probs = preds.squeeze().cpu().detach().numpy()
        return text_id, probs, start, end, placeholder

    def tokenize_and_do_ner(self, *features):
        text1, text2, placeholders = features
        # TODO: now working with text2 assuming that shuffle_texts=True, can cause errors otherwise
        texts = text2
        queries = text1
        # print("features", features)
        # print("texts", texts)
        # print("queries", queries)
        # print("placeholders", placeholders)
        encoded = self._tokenize_with_adjust(texts, queries, placeholders)
        # print("placeholders", placeholders)
        ner_out_raw = self.ner_model(placeholders)[1]
        # print("plain ner_out", ner_out)
        # print(ner_out.shape)
        ner_out_mean = ner_out_raw.mean(axis=1)
        ner_out_max = ner_out_raw.max(axis=1)
        ner_out_first = ner_out_raw[:, 0, :].squeeze()
        ner_out = np.concatenate([ner_out_mean, ner_out_max, ner_out_first], axis=1)
        # print("ner_out after", ner_out.shape)
        return encoded, ner_out

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

            # start_adjust_text_token = max(start_of_placeholder - tokens_left // 2, 0)
            #     end_adjust_text_token = min(end_of_placeholder + tokens_left // 2, self.maxlen - 1)
            # if start_of_placeholder - tokens_left // 2 < 0:
            #     end_adjust_text_token = start_adjust_text_token + tokens_left
            # else:
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