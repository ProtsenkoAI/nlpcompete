from torch.utils import data as torch_data
import transformers


class EvalDataset(torch_data.Dataset):
    def __init__(self, container, mname, has_answers=True):
        self.has_answers = has_answers
        data = container.get_data()
        self.samples = self._get_samples(data)


    def _get_samples(self, data):
        samples = []
        for text, questions in data:
            for question, question_data in questions:
                sample = {"text": text, "question": question}
                if self.has_answers:
                    answers_idxs = question_data["answers"]
                    sample["valid_answers"] = answers_idxs

                samples.append(sample)

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = (sample["text"], sample["question"])
        if self.has_answers:
            return features, sample["valid_answers"]
        return features
        
    def __len__(self):
        return len(self.samples)