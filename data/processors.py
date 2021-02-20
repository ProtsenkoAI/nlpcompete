import transformers
import torch


class QADataProcessor:
    # TODO: maybe move converting from batches to categories from loaders to processor
    # TODO: delete samples whose answers aren't in context
    def __init__(self, mname):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.maxlen = 512

    def preprocess_features_and_labels(self, features, labels, device=None):
        tokenized = self._tokenize(*features)
        labels_in_token_format = self._token_idxs_from_char_idxs(labels, tokenized)
        features = self._features_from_tokenized(tokenized, device)
        labels_proc = self._preproc_labels(labels_in_token_format, device)
        return features, labels_proc

    def preprocess_features(self, features, device=None):
        # proc_features = self._preproc_texts(contexts, questions)
        tokenized = self._tokenize(*features)
        proc_features = self._features_from_tokenized(tokenized, device)
        return proc_features

    def postprocess_preds(self, preds):
        start_logits, end_logits = preds
        start_argmax = torch.argmax(start_logits, dim=-1)
        end_argmax = torch.argmax(end_logits, dim=-1) + 1
        conved = start_argmax.cpu().detach().numpy(), end_argmax.cpu().detach().numpy()
        conved_grouped_by_sample = list(zip(*conved))
        return conved_grouped_by_sample

    def text_from_preds(self, postproc_preds, features):
        # TODO: at the moment we tokenize text two times, have to postproc preds and make predictions
        # in one function
        # TODO: handle case if predictions are out of text
        texts, questions = features
        # tokenized_texts =

    def postprocess_labels(self, labels):
        start, end = labels
        return start.cpu().detach().numpy(), end.cpu().detach().numpy()

    def _tokenize(self, *texts):
        encoded = self.tokenizer(*texts,
                                max_length=self.maxlen,
                                padding="max_length",
                                truncation="longest_first",
                                return_tensors="pt",
                                return_offsets_mapping=True)

        return encoded

    def _token_idxs_from_char_idxs(self, char_idxs, tokenizer_out):
        start_chars, end_chars = char_idxs
        tokens_positions = tokenizer_out["offset_mapping"]
        start_tokens, end_tokens = [], []
        for start_idx, end_idx, text_positions in zip(start_chars, end_chars, tokens_positions):
            text_token_starts, text_token_ends = zip(*text_positions)
            if end_idx <= max(text_token_ends):
                nearest_start = min(text_token_starts, key=lambda x:abs(x-start_idx))
                nearest_end = min(text_token_ends, key=lambda x:abs(x-end_idx))
                token_start = text_token_starts.index(nearest_start)
                token_end = text_token_ends.index(nearest_end)
            else:
                # answer not in context
                token_start, token_end = 0, 0 # TODO: fix it filtering samples like these
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
