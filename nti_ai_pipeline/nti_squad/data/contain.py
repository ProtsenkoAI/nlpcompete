from typing import List

from pipeline.data import BaseContainer
from .types_raw import TrainQuestion, TrainAnswer
from .types_parsed import ParsedAnswer, ParsedQuestionAnswers, ParsedParagraph

import json


class QADataContainer(BaseContainer):
    def __init__(self, path: str):
        self.path = path
        super().__init__()

    def collect_data(self) -> List[ParsedParagraph]:
        data = self._parse_file(self.path)
        return data

    def _parse_file(self, path: str) -> List[ParsedParagraph]:
        texts_and_questions: List[ParsedParagraph] = []
        data_lines = self._read_data_lines(path)
        for line in data_lines:
            text, questions_info = self._get_text_and_questions_info(line)
            texts_and_questions.append((text, questions_info))
        return texts_and_questions

    def _read_data_lines(self, pth):
        with open(pth) as data:
            data_lines = json.load(data)["paragraphs"]
            return data_lines

    def _get_text_and_questions_info(self, line):
        text = line["context"]
        questions_and_answers = []
        for question_with_answers in line["qas"]:
            question = question_with_answers["question"]
            answers = self._get_answers(question_with_answers)
            quest_id = question_with_answers["id"]

            question_data = ParsedQuestionAnswers(id=quest_id, answers=answers)
            questions_and_answers.append((question, question_data))
        return text, questions_and_answers

    def _get_answers(self, qna: TrainQuestion) -> List[ParsedAnswer]:
        answers = []
        for answer in qna["answers"]:
            txt = self._parse_answer(answer)
            answers.append(txt)

        return answers

    def _parse_answer(self, answer: TrainAnswer) -> ParsedAnswer:
        answer_length = len(answer["text"])
        start_idx = answer["answer_start"]
        end_idx = start_idx + answer_length
        return start_idx, end_idx
