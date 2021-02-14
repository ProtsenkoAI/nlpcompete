from torch.utils import data as torch_data
import transformers


class EvalDataset(torch_data.Dataset):
    def __init__(self, container, mname, has_answers=True):
        self.has_answers = has_answers
        self.maxlen = 512
        data = container.get_data()
        self.samples = self._get_samples(data)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)


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
        features = self._preproc_texts(sample["text"], sample["question"])
        if self.has_answers:
            return features, sample["valid_answers"]
        return features

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