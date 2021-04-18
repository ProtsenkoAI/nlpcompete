import json
import os
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Collection, List

from pipeline.modeling import ModelManager
from pipeline.submitting import Submitter


class QASubmitter(Submitter):
    SamplePrediction = Tuple[int, str]   # TODO: check if it's correct
    SubmObj = dict

    def pred_batch(self, batch, model_manager: ModelManager) -> List[SamplePrediction]:
        quest_ids, contexts, questions = batch
        pred_tokens = model_manager.predict_postproc((contexts, questions))

        assert(len(quest_ids) == len(pred_tokens))
        return list(zip(quest_ids, pred_tokens))

    def form_submission(self, preds: List[SamplePrediction]) -> SubmObj:
        return dict(preds)

    def write_submission(self, subm_obj: SubmObj, sub_path: str):
        with open(sub_path, "w", encoding="ascii") as f:
            json.dump(subm_obj, f)
