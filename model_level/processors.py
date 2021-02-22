import transformers
import torch
import numpy as np


class QADataProcessor:
    # TODO: convert logits to probabilities?
    # TODO: maybe move converting from batches to categories from loaders to processor (wow, sounds cool
    #  and we don't need loadercreator anymore)
    # TODO: delete samples whose answers aren't in context
    def __init__(self, mname):
        self.mname = mname
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.maxlen = 512

    def get_init_kwargs(self):
        return {"mname": self.mname}

    def preprocess(self, features, labels=None, device=None):
        tokenized = self._tokenize(*features)
        features = self._features_from_tokenized(tokenized, device)
        if not labels is None:
            labels_in_token_format = self._char_idxs_to_token_idxs(labels, tokenized)
            labels_proc = self._preproc_labels(labels_in_token_format, device)
            return features, labels_proc
        return features

    def postprocess(self, preds, labels=None):
        start_logits, end_logits = preds
        start_arr, end_arr = start_logits.cpu().detach().numpy(), end_logits.cpu().detach().numpy()
        best_start_idxs, best_end_idxs = self._get_best_preds_starts_ends(start_arr, end_arr)
        conved_grouped_by_sample = list(zip(best_start_idxs,
                                            best_end_idxs))

        if not labels is None:
            start, end = labels
            start_and_end = start.cpu().detach().numpy(), end.cpu().detach().numpy()
            zipped = list(zip(*start_and_end))
            return conved_grouped_by_sample, zipped
        return conved_grouped_by_sample

    def text_from_token_idxs(self, token_idxs, features):
        # TODO: at the moment we tokenize text two times, have to postproc preds and make predictions in one function
        texts, questions = features
        offset_mapping = self._tokenize(list(texts))["offset_mapping"]
        answers = []
        for orig_text, (start, end), mapping in zip(texts, token_idxs, offset_mapping):
            # TODO: at the moment if answer start and end are in [PAD] part, then we return full text, should fix it somehow
            answer_start_char = mapping[int(start), 0]
            answer_end_char = mapping[int(end), 1]
            if answer_end_char == 0:
                answer_end_char = len(orig_text)
            answer = orig_text[answer_start_char: answer_end_char]

            answers.append(answer)
        return answers

    def _tokenize(self, *texts):
        encoded = self.tokenizer(*texts,
                                 max_length=self.maxlen,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt",
                                 return_offsets_mapping=True)

        return encoded

    def _char_idxs_to_token_idxs(self, char_idxs, tokenizer_out):
        # TODO: refactor and fix
        start_chars, end_chars = char_idxs
        tokens_positions = tokenizer_out["offset_mapping"]
        start_tokens, end_tokens = [], []
        for start_idx, end_idx, text_positions in zip(start_chars, end_chars, tokens_positions):
            text_token_starts, text_token_ends = zip(*text_positions)
            if end_idx <= max(text_token_ends):
                nearest_start = min(text_token_starts, key=lambda x: abs(x - start_idx))
                nearest_end = min(text_token_ends, key=lambda x: abs(x - end_idx))
                token_start = text_token_starts.index(nearest_start)
                token_end = text_token_ends.index(nearest_end)
            else:
                # answer not in context
                token_start, token_end = 0, 0  # TODO: fix it filtering samples like these
            start_tokens.append(token_start)
            end_tokens.append(token_end)

        return start_tokens, end_tokens

    def _preproc_labels(self, labels, device):
        labels_proc = self._create_tensors(labels, device)
        return labels_proc

    def _features_from_tokenized(self, tokenizer_out, device):
        # maybe later we'll need a tokenizer wrapper to return this parts
        needed_parts = [tokenizer_out["input_ids"],
                        tokenizer_out["attention_mask"],
                        tokenizer_out["token_type_ids"]
                        ]
        tensors = self._create_tensors(needed_parts, device)
        return tensors

    def _create_tensors(self, arrays, device):
        tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
            if not device is None:
                arr = arr.to(device, copy=True)
            tensors.append(arr)
        return tensors

    def _get_best_preds_starts_ends(self, start_logits, end_logits):
        """
        Finds best start and end token idxs, whose sum of probabilities is maximized (check BERT paper for more
        info).
        """
        maximums = np.zeros(len(start_logits))
        best_start_idxs = np.zeros(len(start_logits))
        best_end_idxs = np.zeros(len(start_logits))

        for token_idx in range(start_logits.shape[1]):
            token_start_probs = start_logits[:, token_idx]
            right_end_probs = end_logits[:, token_idx:]
            right_argmaxes = np.argmax(right_end_probs, axis=-1) + token_idx
            right_max_vals = np.max(right_end_probs)
            curr_sum_prob = token_start_probs + right_max_vals
            best_start_idxs = np.where(curr_sum_prob > maximums, token_idx, best_start_idxs)
            best_end_idxs = np.where(curr_sum_prob > maximums, right_argmaxes, best_end_idxs)
            maximums = np.maximum(maximums, curr_sum_prob)
        return best_start_idxs, best_end_idxs
