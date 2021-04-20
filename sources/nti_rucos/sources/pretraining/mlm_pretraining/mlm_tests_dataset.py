from torch.utils import data as torch_data
import torch


class MLMTextsDataset(torch_data.Dataset):
    def __init__(self, text_samples, tokenizer, max_tokens_num=512):
        self.max_tokens_num = max_tokens_num
        self.all_train_texts = []

        self.tokenizer = tokenizer
        for text_data in text_samples:
            self.all_train_texts.append(text_data.text1)

    def __getitem__(self, idx):
        text = self.all_train_texts[idx]
        ids = self.tokenizer(text, add_special_tokens=True, truncation=True,
                             max_length=self.max_tokens_num)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.all_train_texts)
