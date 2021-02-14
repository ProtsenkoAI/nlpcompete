import json
import os


class Submitter:
    def __init__(self, container, subm_dir):
        self.container = container
        self.subm_dir = subm_dir
        os.makedirs(subm_dir, exist_ok=True)

    def create_submission(self, subm_file_name, answers_idxs):
        answers = self._get_answers_by_idxs(answers_idxs)
        data = self.container.get_data()
        ids = self._get_data_ids(data)

        subm_dict = self._form_submission(ids, answers)
        self._write_subm_json(subm_file_name, subm_dict)

    def _get_answers_by_idxs(self, answers_idxs):
        answers = []
        data = self.container.get_data()
        answers_idx = 0

        for text, questions in data:
            questions_nb = len(questions)
            for ans_start, ans_end in answers_idxs[answers_idx: answers_idx + questions_nb]:
                answer = text[ans_start: ans_end]
                answers.append(answer)
            answers_idx += questions_nb
            
        return answers

    def _get_data_texts(self, data):
        return data["text"]

    def _get_data_ids(self, data):
        ids = []
        for text, text_data in data:
            for question, question_data in text_data:
                question_id = question_data["id"]
                ids.append(question_id)
        return ids

    def _form_submission(self, ids, answers):
        return dict(zip(ids, answers))

    def _write_subm_json(self, file_name, data):
        subm_path = os.path.join(self.subm_dir, file_name + ".json")
        with open(subm_path, "w", encoding="ascii") as f:
            json.dump(data, f)
