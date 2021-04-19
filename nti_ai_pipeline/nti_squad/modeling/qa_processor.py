import torch
from typing import Union

from .qa_proc_assistant import QAProcAssistant
from pipeline.modeling import BaseProcessor
from pipeline.modeling.types import BatchWithLabels, BatchWithoutLabels
from .types import UnprocLabels, UnprocFeatures, ProcLabels, TrueTokenIdxs, SubmSamplePredWithProbs,\
    SubmBatchPredWithProbs, SubmBatchPred


class QADataProcessor(BaseProcessor):
    # TODO: refactor (esp. working with probs in postprocess)
    def __init__(self, mname: str, max_answer_token_len=50):
        self.mname = mname
        self.max_answer_token_len = max_answer_token_len
        self.proc_assistant = QAProcAssistant(mname, max_answer_token_len)

    def get_init_kwargs(self) -> dict:
        return {"mname": self.mname,
                "max_answer_token_len": self.max_answer_token_len}

    def preprocess(self, features: UnprocFeatures, labels: Union[None, UnprocLabels] = None, device: torch.device = None
                   ) -> Union[BatchWithoutLabels, BatchWithLabels]:
        tokenized = self.proc_assistant.tokenize(*features)
        if labels is not None:
            tokenized, labels = self.proc_assistant.filter_samples(tokenized, labels)

        features_proc = self.proc_assistant.features_from_tokenized(tokenized, device)
        if labels is not None:
            labels_in_token_format = self.proc_assistant.char_idxs_to_token_idxs(labels, tokenized)
            labels_proc = self._preproc_labels(labels_in_token_format, device)
            return BatchWithLabels(features_proc, labels_proc)
        return BatchWithoutLabels(features_proc)

    def after_forward(self, raw_model_out):
        start_logits, end_logits = raw_model_out.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def postprocess(self, preds, src_features, src_labels=None, return_probs=False) -> Union[SubmBatchPredWithProbs, SubmBatchPred]:
        # TODO: new-style processor should return list of samples
        src_texts, src_questions = src_features
        tokenized = self.proc_assistant.tokenize(src_texts, src_questions)
        if src_labels is not None:
            tokenized, src_labels, (src_texts,) = self.proc_assistant.filter_samples(tokenized, src_labels, [src_texts])
        preds = self.proc_assistant.fill_zeros_out_of_context(preds, tokenized)
        start_logits, end_logits = preds
        out = self.proc_assistant.get_best_preds_starts_ends(start_logits, end_logits,
                                                             return_probs=return_probs)
        if return_probs:
            (best_start_idxs, best_end_idxs), best_probs = out
        else:
            best_start_idxs, best_end_idxs = out
        final_pred = self.proc_assistant.text_from_token_idxs(best_start_idxs, best_end_idxs, src_texts, tokenized)
        if return_probs:
            # final_pred = SubmPredWithProbs(final_pred, best_probs)
            final_pred = [SubmSamplePredWithProbs(pred_text, prob) for pred_text, prob in zip(final_pred, best_probs)]
        else:
            final_pred = [pred_text for pred_text in final_pred]

        if src_labels is not None:
            start_idxs, end_idxs = src_labels
            ground_truth_text = self.proc_assistant.crop_text_by_idxs(src_texts, start_idxs, end_idxs)
            # return BatchWithLabels(final_pred, ground_truth_text)
            return list(zip(final_pred, ground_truth_text))
        return BatchWithoutLabels(final_pred)

    def _preproc_labels(self, labels: TrueTokenIdxs, device) -> ProcLabels:
        labels_proc = self.proc_assistant.create_tensors(labels, device)
        assert len(labels_proc) == 2
        return tuple(labels_proc)
