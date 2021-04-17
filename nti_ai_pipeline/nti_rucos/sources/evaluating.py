from typing import Tuple
import numpy as np
import pandas as pd

from pipeline.modeling import ModelManager
from pipeline.training import Validator


class RucosValidator(Validator):
    AllPreds = pd.DataFrame
    AllLabels = pd.DataFrame
    Preds = pd.DataFrame
    Labels = pd.DataFrame

    def __init__(self, tqdm_mode="notebook"):
        # TODO: metrics are hardcoded now, maybe later we'll need to get them in arguments
        self.metric = self._score_f1_from_df

        self.preds_df = pd.DataFrame(columns=["id", "prob"])
        self.labels_df = pd.DataFrame(columns=["labels"])
        super().__init__(tqdm_mode)

    def pred_batch(self, manager: ModelManager, batch) -> Tuple[Preds, Labels]:
        features, labels = batch
        question_idx, *features_to_preproc_forward = features
        preds, proc_labels = manager.preproc_forward(features_to_preproc_forward, labels)
        preds_df = pd.DataFrame({"id": question_idx,
                                 "prob": preds.cpu().detach().numpy()[:, 1]
                                 })
        labels_df = pd.DataFrame({"label": proc_labels.cpu().detach().numpy()})
        return preds_df, labels_df

    def store_batch_res(self, preds, labels):
        self.preds_df = self.preds_df.append(preds, ignore_index=True)
        self.labels_df = self.labels_df.append(labels, ignore_index=True)

    def get_all_preds_and_labels(self) -> Tuple[AllPreds, AllLabels]:
        return self.preds_df, self.labels_df

    def calc_metric(self, preds: AllPreds, labels: AllLabels) -> float:
        preds_and_labels = pd.concat([preds, labels], axis=1)
        return self.metric(preds_and_labels)

    def clear_accumulated_preds_and_labels(self):
        self.preds_df = pd.DataFrame(columns=["id", "prob"])
        self.labels_df = pd.DataFrame(columns=["labels"])

    def _score_f1_from_df(self, df):
        text_groups = df.groupby("id")
        sample_f1s = []
        for _, group in text_groups:
            best_candidate = group.sort_values("prob", ascending=False).iloc[0]
            f1_sample = best_candidate["label"]  # if best candidate is labeled 1, then f1 of this sample is 1
            sample_f1s.append(f1_sample)
        return float(np.mean(sample_f1s))
