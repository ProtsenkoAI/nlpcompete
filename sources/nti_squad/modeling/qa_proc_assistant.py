import transformers
import torch
from typing import List
import numpy as np

from .types import ProcFeatures


class QAProcAssistant:
    def __init__(self, mname, max_answer_token_len=50):
        self.mname = mname
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.maxlen = 512
        self.max_answer_token_len = max_answer_token_len

    def features_from_tokenized(self, tokenizer_out, device) -> ProcFeatures:
        # maybe later we'll need a tokenizer wrapper to return this parts
        needed_parts = [tokenizer_out["input_ids"],
                        tokenizer_out["attention_mask"],
                        tokenizer_out["token_type_ids"]
                        ]
        tensors = self.create_tensors(needed_parts, device)
        assert len(tensors) == 3
        return tuple(tensors)

    def tokenize(self, *texts):
        encoded = self.tokenizer(*texts,
                                 max_length=self.maxlen,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="np",
                                 return_offsets_mapping=True)

        return encoded

    def get_best_preds_starts_ends(self, start_logits, end_logits, return_probs=False):
        """
        Finds best start and end token idxs, whose sum of probabilities is maximized (check BERT paper for more
        info).
        """
        start_logits, end_logits = start_logits.cpu().detach().numpy(), end_logits.cpu().detach().numpy()
        maximums = np.zeros(len(start_logits))
        best_start_idxs = np.zeros(len(start_logits))
        best_end_idxs = np.zeros(len(start_logits))

        for token_idx in range(start_logits.shape[1]):
            token_start_probs = start_logits[:, token_idx]
            right_end_probs = end_logits[:, token_idx:token_idx + self.max_answer_token_len]
            right_argmaxes = np.argmax(right_end_probs, axis=-1) + token_idx
            right_max_vals = np.max(right_end_probs)
            curr_sum_prob = token_start_probs + right_max_vals
            best_start_idxs = np.where(curr_sum_prob > maximums, token_idx, best_start_idxs)
            best_end_idxs = np.where(curr_sum_prob > maximums, right_argmaxes, best_end_idxs)
            maximums = np.maximum(maximums, curr_sum_prob)
        if return_probs:
            return (best_start_idxs, best_end_idxs), maximums
        return best_start_idxs, best_end_idxs

    def create_tensors(self, arrays, device) -> List[torch.Tensor]:
        tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
            if not device is None:
                arr = arr.to(device, copy=True)
            tensors.append(arr)
        return tensors

    def fill_zeros_out_of_context(self, predictions, tokenized):
        start_preds, end_preds = predictions
        # start_preds = self._to_numpy(start_preds)
        # end_preds = self._to_numpy(end_preds)

        for text_idx, text_token_types in enumerate(tokenized["token_type_ids"]):
            question_start_idx = list(text_token_types).index(1)
            start_preds[text_idx, question_start_idx:] = 0
            end_preds[text_idx, question_start_idx:] = 0
        return start_preds, end_preds

    def _to_numpy(self, tensor: torch.Tensor) -> np.array:
        return tensor.detach().numpy()

    def filter_samples(self, tokenized, labels, other_arrays=()):
        """
        :param other_arrays: len(other_arrays !elems!) == len(tokenized).
        :return:
        """
        is_in_context = self._check_answer_span_is_in_context(tokenized, labels)
        filtered_labels = []
        for label_categ in labels:
            filt_categ = np.array(label_categ)[is_in_context]
            filtered_labels.append(filt_categ)

        filtered_tokenized = {}
        for key, val in tokenized.items():
            filtered_val = val[is_in_context]
            filtered_tokenized[key] = filtered_val

        filtered_other_arrays = []
        for other_arr in other_arrays:
            other_arr_filt = np.array(other_arr)[is_in_context]
            filtered_other_arrays.append(other_arr_filt)
        if len(filtered_other_arrays):
            return filtered_tokenized, filtered_labels, filtered_other_arrays
        return filtered_tokenized, filtered_labels

    def _check_answer_span_is_in_context(self, tokenized, labels) -> np.array:
        text_end_chars = tokenized["offset_mapping"][:, :, 1].max(axis=1)
        label_start_chars, label_end_chars = labels
        is_in_context = np.array(label_end_chars) < text_end_chars
        return is_in_context

    def crop_text_by_idxs(self, texts, starts, ends):
        cropped = []

        assert(len(texts) == len(starts) and len(starts) == len(ends))
        for text, start, end in zip(texts, starts, ends):
            cropped.append(text[start: end])
        return cropped

    def text_from_token_idxs(self, token_starts, token_ends, texts, tokenized):
        start_chars, end_chars = self.token_idxs_to_to_char_idxs(token_starts, token_ends, tokenized)
        answers = []
        assert(len(start_chars) == len(end_chars) and len(end_chars) == len(texts))
        for (start, end, orig_text) in zip(start_chars, end_chars, texts):
            answer = orig_text[start: end]
            answers.append(answer)

        return answers

    def token_idxs_to_to_char_idxs(self, token_starts, token_ends, tokenized):
        token_idxs = list(zip(token_starts, token_ends))
        offset_mapping = tokenized["offset_mapping"]
        start_chars, end_chars = [], []
        if len(token_idxs) != len(offset_mapping):
            raise ValueError(f"token_idxs {token_idxs} offset_mapping {offset_mapping}")
        for (start, end), mapping in zip(token_idxs, offset_mapping):
            answer_start_char = mapping[int(start), 0]
            answer_end_char = mapping[int(end), 1]
            start_chars.append(answer_start_char)
            end_chars.append(answer_end_char)
        return start_chars, end_chars

    def char_idxs_to_token_idxs(self, char_idxs, tokenizer_out):
        start_chars, end_chars = char_idxs
        tokens_positions = tokenizer_out["offset_mapping"]
        start_tokens, end_tokens = [], []

        assert(len(start_chars) == len(end_chars) and len(end_chars) == len(tokens_positions))
        for start_idx, end_idx, text_positions in zip(start_chars, end_chars, tokens_positions):
            text_token_starts, text_token_ends = zip(*text_positions)
            if end_idx <= max(text_token_ends):
                nearest_start = min(text_token_starts, key=lambda x: abs(x - start_idx))
                nearest_end = min(text_token_ends, key=lambda x: abs(x - end_idx))
                token_start = text_token_starts.index(nearest_start)
                token_end = text_token_ends.index(nearest_end)
            else:
                # answer not in context
                raise ValueError(f"end_idx is wrong: {end_idx, max(text_token_ends)}")
            start_tokens.append(token_start)
            end_tokens.append(token_end)

        return start_tokens, end_tokens
