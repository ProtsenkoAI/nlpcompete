from torch.utils import data as torch_data
import torch


class SubmDataset(torch_data.Dataset):
    def __init__(self, container):
        data = container.get_data()
        self.samples = self._get_samples(data)

    def _get_samples(self, data):
        samples = []
        for text, questions in data:
            for question, question_data in questions:
                quest_id = question_data["id"]
                sample = {"id": quest_id, "text": text, "question": question}
                samples.append(sample)

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = (sample["id"], sample["text"], sample["question"])
        return features

    def __len__(self):
        return len(self.samples)
