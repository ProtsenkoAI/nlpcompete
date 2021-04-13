from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pipeline.modeling import ModelManager
from pipeline.training import Validator


class RucosValidator(Validator):
    def __init__(self):
        # TODO: metrics are hardcoded now, maybe later we'll need to get them in arguments
        self.metric = self._f1_score

    def eval(self, manager: ModelManager, test_loader: DataLoader) -> float:
        metric_val = self.metric(manager, test_loader)
        print("Eval value:", metric_val)
        return metric_val

    def _f1_score(self, manager, test: DataLoader):
        val_df = pd.DataFrame(columns=["id", "prob", "label"])
        for features, labels in tqdm(test, mininterval=1):  # TODO: use get_tqdm from util
            question_idx, *features_to_preproc_forward = features
            preds, proc_labels = manager.preproc_forward(features_to_preproc_forward, labels)
            tmp_df = pd.DataFrame({"id": question_idx,
                                   "prob": preds.cpu().detach().numpy()[:, 1],
                                   "label": proc_labels.cpu().detach().numpy()
                                   })
            val_df = val_df.append(tmp_df, ignore_index=True)

        score = self._score_f1_from_df(val_df)
        return score

    def _score_f1_from_df(self, df):
        text_groups = df.groupby("id")
        sample_f1s = []
        for _, group in text_groups:
            best_candidate = group.sort_values("prob", ascending=False).iloc[0]
            f1_sample = best_candidate["label"]  # if best candidate is labeled 1, then f1 of this sample is 1
            sample_f1s.append(f1_sample)
        return float(np.mean(sample_f1s))
