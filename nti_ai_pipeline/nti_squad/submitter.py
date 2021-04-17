import json
import os
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Collection

from pipeline.modeling import ModelManager


class Submitter:
    # TODO: inherit from base
    def __init__(self, subm_dir="./"):
        self.subm_dir = subm_dir
        os.makedirs(subm_dir, exist_ok=True)

    def create_submission(self, model_manager: ModelManager, loader: DataLoader, subm_file_name="submission") -> None:
        ids, answers = self._get_question_ids_make_answers(loader, model_manager)
        subm_dict = self._form_submission(ids, answers)
        self._write_subm_json(subm_file_name, subm_dict)

    def _get_question_ids_make_answers(self, loader, manager) -> Tuple[Collection[int], Collection[str]]:
        ids, answers = [], []
        for features in tqdm(loader, desc="Making submit predictions"):
            quest_ids, contexts, questions = features
            pred_tokens = manager.predict_postproc((contexts, questions))

            ids += list(quest_ids)
            answers += pred_tokens
        return ids, answers

    def _form_submission(self, ids: Collection[int], answers: Collection[str]):
        assert(len(ids) == len(answers))
        return dict(zip(ids, answers))

    def _write_subm_json(self, file_name, data):
        subm_path = os.path.join(self.subm_dir, file_name + ".json")
        with open(subm_path, "w", encoding="ascii") as f:
            json.dump(data, f)
