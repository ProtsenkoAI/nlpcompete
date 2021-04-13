import torch

from legacy.model_level.processors.qa_proc_assistant import QAProcAssistant

from legacy.model_level.qa_types import *


class QADataProcessor:
    # TODO: the class is too large, maybe add assistant components
    def __init__(self, mname: str, max_answer_token_len=50):
        self.mname = mname
        self.max_answer_token_len = max_answer_token_len
        self.proc_assistant = QAProcAssistant(mname, max_answer_token_len)

    def get_init_kwargs(self) -> dict:
        return {"mname": self.mname,
                "max_answer_token_len": self.max_answer_token_len}

    def preprocess(self, features: UnprocFeatures, labels: Union[None, UnprocLabels]=None, device: torch.device=None
                   ) -> Union[ProcFeatures, Tuple[ProcFeatures, ProcLabels]]:
        tokenized = self.proc_assistant.tokenize(*features)
        if not labels is None:
            tokenized, labels = self.proc_assistant.filter_samples(tokenized, labels)

        features_proc = self.proc_assistant.features_from_tokenized(tokenized, device)
        if not labels is None:
            labels_in_token_format = self.proc_assistant.char_idxs_to_token_idxs(labels, tokenized)
            labels_proc = self._preproc_labels(labels_in_token_format, device)
            return features_proc, labels_proc
        return features_proc

    def postprocess(self, preds, src_features, src_labels=None):
        src_texts, src_questions = src_features
        tokenized = self.proc_assistant.tokenize(src_texts, src_questions)
        if not src_labels is None:
            tokenized, src_labels, (src_texts,) = self.proc_assistant.filter_samples(tokenized, src_labels, [src_texts])
        preds = self.proc_assistant.fill_zeros_out_of_context(preds, tokenized)
        start_logits, end_logits = preds
        best_start_idxs, best_end_idxs = self.proc_assistant.get_best_preds_starts_ends(start_logits, end_logits)
        pred_text = self.proc_assistant.text_from_token_idxs(best_start_idxs, best_end_idxs, src_texts, tokenized)

        if not src_labels is None:
            start_idxs, end_idxs = src_labels
            ground_truth_text = self.proc_assistant.crop_text_by_idxs(src_texts, start_idxs, end_idxs)
            return pred_text, ground_truth_text
        return pred_text

    def _preproc_labels(self, labels: TrueTokenIdxs, device) -> ProcLabels:
        labels_proc = self.proc_assistant.create_tensors(labels, device)
        assert len(labels_proc) == 2
        return tuple(labels_proc)