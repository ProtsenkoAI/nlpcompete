from tqdm.notebook import tqdm
import numpy as np


class Validator:
    def __init__(self, loader_builder):
        # TODO: metrics are hardcoded now, maybe later we'll need to get them in arguments  
        # TODO: add multiple metrics
        self.metric = self._f1_qa_score
        self.loader_builder = loader_builder

    def eval(self, manager, dataset):
        manager.get_model().eval()
        test_loader = self.loader_builder.build(dataset)
        preds, labels = self._pred_all_batches(manager, test_loader)
        metric_val = self.metric(preds, labels)
        print(f"Example of eval probs: {preds[:20]}")
        print("Eval value:", metric_val)
        return metric_val

    def _pred_all_batches(self, manager, test):
        all_preds = []
        all_labels = []
        for batch in tqdm(test, desc="eval"):
            features, labels_unproc = batch
            preds, labels_proc = manager.predict_postproc_labeled(features, labels_unproc)
            all_preds += list(preds)
            all_labels += list(labels_proc)
        return all_preds, all_labels
        

    def _em_score(self, preds, correct_answers):
        raise NotImplementedError

    def _f1_qa_score(self, preds, labels):
        """F1 metric for QA task. The original SQUAD implementation is being used, but here we work with indexes, 
        not tokens. 
        Source code: https://github.com/nlpyang/pytorch-transformers/blob/master/examples/utils_squad_evaluate.py"""
        # TODO: documentation
        # TODO: fix to use number of tokens, not letters!
        raise NotImplementedError
        
        samples_f1 = []
        for sample_labels, sample_preds in zip(labels, preds):
            f1_of_sample = self._sample_f1(*sample_labels, *sample_preds)
            samples_f1.append(f1_of_sample)
        return np.mean(samples_f1)

    def _sample_f1(self, gold_ans_start, gold_ans_end, pred_start, pred_end):
        most_left_end = min(gold_ans_end, pred_end)
        most_right_start = max(gold_ans_start, pred_start)
        num_same = most_left_end - most_right_start

        if num_same <= 0:
            return 0

        gold_ans_len = gold_ans_end - gold_ans_start
        pred_len = pred_end - pred_start
        precision = 1.0 * num_same / pred_len
        recall = 1.0 * num_same / gold_ans_len
        if precision + recall == 0:
            return 0

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

