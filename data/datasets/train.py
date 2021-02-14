from torch.utils import data as torch_data
import torch
import transformers


class TrainDataset(torch_data.Dataset):
    # TODO: move tokenization to another place
    def __init__(self, container, mname):
        data = container.get_data()
        self.samples = self._get_samples(data)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)

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
        context, question = (sample["text"], sample["question"])
        features = self._preproc_texts(context, question)
        answer_start_end = sample["answer_start"], sample["answer_end"]
        tensor_labels = torch.tensor(answer_start_end).float()
        return features, tensor_labels

    def _preproc_texts(self, *texts):
        encoded = self.tokenizer(*texts,
                                max_length=self.maxlen,
                                padding="max_length",
                                truncation="longest_first",
                                return_tensors="pt")

        encoded_proc = self._preprocess_tokenizer_out(encoded)

        return [encoded_proc["input_ids"], 
                encoded_proc["attention_mask"], 
                encoded_proc["token_type_ids"]
               ]

    def _preprocess_tokenizer_out(self, out):
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.squeeze(0)

        return out

    def __len__(self):
        return len(self.samples)
