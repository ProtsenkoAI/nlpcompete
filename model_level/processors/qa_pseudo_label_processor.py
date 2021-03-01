import torch
from typing import Tuple, List, Union

from .qa_proc_assistant import QAProcAssistant


UnprocFeatures = Tuple[List[str], List[str]]
ProcFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class QAPseudoLabelProcessor:
    # TODO: open only preprocess and postprocess
    def __init__(self, mname: str, max_answer_token_len=50):
        self.proc_assistant = QAProcAssistant(mname, max_answer_token_len)

    def preprocess(self, features: UnprocFeatures, labels=None, device: torch.device=None
                   ) -> ProcFeatures:
        self._check_labels_none(labels)
        tokenized = self.proc_assistant.tokenize(*features)
        features_proc = self.proc_assistant.features_from_tokenized(tokenized, device)
        return features_proc

    def postprocess(self, preds, features, labels=None):
        self._check_labels_none(labels)
        (start_token_idxs, end_token_idxs), pred_probs = self._best_candidates(preds)
        start_char_idxs, end_char_idxs = self._token_start_end_to_char_idxs(start_token_idxs, end_token_idxs, features)
        return (start_char_idxs, end_char_idxs), pred_probs

    def _best_candidates(self, model_preds):
        (best_start_idxs, best_end_idxs), pred_probs = self.proc_assistant.get_best_preds_starts_ends(*model_preds,
                                                                                                      return_probs=True)
        return (best_start_idxs, best_end_idxs), pred_probs

    def _token_start_end_to_char_idxs(self, start_tokens, end_tokens, features):
        texts, questions = features
        tokenized = self.proc_assistant.tokenize(list(texts))
        start_chars, end_chars = self.proc_assistant.token_idxs_to_to_char_idxs(start_tokens, end_tokens, tokenized)
        return start_chars, end_chars

    def _check_labels_none(self, labels):
        if not labels is None:
            raise ValueError("Labels aren't supported in QAPseudoLabelProcessor")