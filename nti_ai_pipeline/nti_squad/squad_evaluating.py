import numpy as np
import collections
from typing import List, Tuple

from pipeline.modeling import ModelManager
from pipeline.training import Validator


class SQuADValidator(Validator):
    AllPreds = List[str]
    AllLabels = List[str]
    Preds = List[str]
    Labels = List[str]

    def __init__(self, tqdm_mode="notebook"):
        # TODO: metrics are hardcoded now, maybe later we'll need to get them in arguments
        self.metric = self._f1_qa_score
        self.preds = []
        self.labels = []
        super().__init__(tqdm_mode)

    def pred_batch(self, manager: ModelManager, batch) -> Tuple[Preds, Labels]:
        preds, labels_proc = manager.predict_postproc(*batch)
        return preds, labels_proc

    def store_batch_res(self, batch_preds: List, labels: List):
        self.preds += batch_preds
        self.labels += labels

    def get_all_preds_and_labels(self) -> Tuple[AllPreds, AllLabels]:
        return self.preds, self.labels

    def calc_metric(self, preds: AllPreds, labels: AllLabels) -> float:
        return self.metric(preds, labels)

    def clear_accumulated_preds_and_labels(self):
        self.preds = []
        self.labels = []

    def _em_score(self, preds, correct_answers):
        raise NotImplementedError

    def _f1_qa_score(self, preds: List[str], true_texts: List[str]) -> float:
        """F1 metric for QA task. The original SQUAD implementation is being used, but here we work with indexes,
        not tokens.
        Source code: https://github.com/nlpyang/pytorch-transformers/blob/master/examples/utils_squad_evaluate.py
        """
        samples_f1 = []
        print("some true texts", true_texts[:20])
        print("some model answers", preds[:20])
        assert(len(true_texts) == len(preds))
        for sample_labels, sample_preds in zip(true_texts, preds):
            f1_of_sample = self._sample_f1(sample_labels, sample_preds)
            samples_f1.append(f1_of_sample)

        return float(np.mean(samples_f1))

    def _sample_f1(self, true_answer: str, predicted: str) -> float:
        pred_toks = predicted.split()
        gold_toks = true_answer.split()
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
