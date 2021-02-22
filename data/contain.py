import json
from typing import Optional, List

from .types.raw import TrainQuestion, TrainAnswer, TrainParagraph
from .types.parsed import ParsedAnswer, ParsedQuestionAnswers, ParsedQuestion, ParsedParagraph


class DataContainer:
    def __init__(self, path: str, nrows: Optional[int] = None):
        self.path = path
        self.nrows = nrows

    def get_data(self) -> List[ParsedParagraph]:
        data = self._parse_file(self.path)
        return data

    def _parse_file(self, path: str) -> List[ParsedParagraph]:
        texts_and_questions: List[ParsedParagraph] = []
        data_lines = self._read_data_lines(path)
        for line in data_lines:
            paragraph = self._get_text_and_questions_info(line)
            texts_and_questions.append(paragraph)
        return texts_and_questions

    def _read_data_lines(self, pth: str) -> List[TrainParagraph]:
        with open(pth) as data:
            data_lines: List[TrainParagraph] = json.load(data)["paragraphs"]
            if self.nrows is not None:
                data_lines = data_lines[:self.nrows]
            return data_lines

    def _get_text_and_questions_info(self, line: TrainParagraph) -> ParsedParagraph:
        text = line["context"]
        questions_and_answers: List[ParsedQuestion] = []
        for question_with_answers in line["qas"]:
            question = question_with_answers["question"]
            answers = self._get_answers(question_with_answers)
            quest_id = question_with_answers["id"]
            
            # question_data = {"id": quest_id, "answers": answers}
            question_data = ParsedQuestionAnswers(id=quest_id, answers=answers)
            # questions_and_answers.append((question, question_data))
            questions_and_answers.append(ParsedQuestion(question, question_data))
        # return text, questions_and_answers
        return ParsedParagraph(text, questions_and_answers)

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
        return ParsedAnswer(start_idx, end_idx)
