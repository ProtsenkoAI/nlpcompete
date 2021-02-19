from torch.utils import data as torch_data
import torch


class StandardDataset(torch_data.Dataset):
    def __init__(self, container, mname, has_answers=True):
        data = container.get_data()
        self.has_answers = has_answers
        self.samples = self._get_samples(data)

        self.maxlen = 512

    def _get_samples(self, data):
        samples = []
        for text, questions in data:
            for question, question_data in questions:
                sample = {"text": text, "question": question}
                if self.has_answers:
                    for answer_idxs in question_data["answers"]:
                        start, end = answer_idxs
                        sample = {"text": text, "question": question, 
                                "answer_start": start, "answer_end": end}
                        samples.append(sample)
                else:
                    samples.append(sample)

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = (sample["text"], sample["question"])
        if self.has_answers:
            answer_start_end = sample["answer_start"], sample["answer_end"]
            return features, answer_start_end
        return features

    def __len__(self):
        return len(self.samples)
