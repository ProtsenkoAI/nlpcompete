import json
import os
from tqdm.notebook import tqdm


class Submitter:
    def __init__(self, loader_builder, subm_dir="./"):
        self.loader_builder = loader_builder
        self.subm_dir = subm_dir
        os.makedirs(subm_dir, exist_ok=True)

    def create_submission(self, model_manager, dataset, subm_file_name="submission"):
        loader = self.loader_builder.build(dataset, has_answers=False)
        ids, answers = self._get_question_ids_make_answers(loader, model_manager)
        subm_dict = self._form_submission(ids, answers)
        self._write_subm_json(subm_file_name, subm_dict)

    def _get_question_ids_make_answers(self, loader, manager):
        ids, answers = [], []
        for features in tqdm(loader, desc="Making submit predictions"):
            quest_ids, contexts, questions = features
            pred_tokens_idxs = manager.predict_postproc((contexts, questions))
            answers = self._get_answers_by_idxs(pred_tokens_idxs, contexts)

            ids += list(quest_ids)
            answers += answers
            print("sub", ids)
            print(answers)
        return ids, answers

    def _get_answers_by_idxs(self, start_end_idxs, texts):
        answers = []
        for (start, end), text in zip(start_end_idxs, texts):
            answer = text[start: end]
            answers.append(answer)
            
        return answers

    def _form_submission(self, ids, answers):
        return dict(zip(ids, answers))

    def _write_subm_json(self, file_name, data):
        subm_path = os.path.join(self.subm_dir, file_name + ".json")
        with open(subm_path, "w", encoding="ascii") as f:
            json.dump(data, f)
