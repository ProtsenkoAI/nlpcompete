from torch.utils import data as torch_data
import torch


class TrainDataset(torch_data.Dataset):
    # TODO: move tokenization to another place
    def __init__(self, container, mname):
        data = container.get_data()
        self.samples = self._get_samples(data)

        self.has_answers = True
        self.maxlen = 512

    def _get_samples(self, data):
        samples = []
        for text, questions in data:
            for question, question_data in questions:
                for answer_idxs in question_data["answers"]:
                    start, end = answer_idxs
                    sample = {"text": text, "question": question, 
                              "answer_start": start, "answer_end": end}
                    samples.append(sample)

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = (sample["text"], sample["question"])
        answer_start_end = sample["answer_start"], sample["answer_end"]
        return features, answer_start_end

    def __len__(self):
        return len(self.samples)
