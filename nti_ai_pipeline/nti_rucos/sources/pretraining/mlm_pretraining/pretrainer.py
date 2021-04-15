import transformers
from torch.utils import data as torch_data
from transformers.data.data_collator import DataCollatorForLanguageModeling
import torch
from glob import glob
import os

from transformers import BertForMaskedLM, BertTokenizer
from pipeline.data import BaseContainer


class MLMBertPretrainer:
    # TODO: at the moment returns the name of latest save, need to check eval metrics
    # TODO: works only with bert, need to broaden use case
    # TODO: move some parts to pipeline
    def __init__(self, cache_dir="./cache_models/", checkpoints_dir="./pretrain_checkpoints"):
        self.cache_dir = cache_dir
        self.checkpoints_dir = checkpoints_dir

    def pretrain(self, mname: str, data_container: BaseContainer,
                 test_size: float = 0.2, nepochs=3, eval_every=20000,
                 save_every=2000, batch_size=4) -> str:
        # TODO: passing tokenizer both in dataset and DataCollator, need to pass only in one case
        """
        Pretrains BERT and returns path to file with its weights
        """
        pre_train_bert, tokenizer = self._get_model_and_tokenizer(mname)

        train_samples, val_samples = data_container.train_test_split(test_size=test_size)
        train_dataset = TextsDataset(train_samples, tokenizer)
        val_dataset = TextsDataset(val_samples, tokenizer)

        self._train_model(pre_train_bert, tokenizer, train_dataset, val_dataset,
                          nepochs=nepochs, eval_every=eval_every, save_every=save_every,
                          batch_size=batch_size)
        return self._get_last_save_name()

    def _get_model_and_tokenizer(self, model_name):
        model = BertForMaskedLM.from_pretrained(model_name, cache_dir=self.cache_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def _train_model(self, model, tokenizer, train_dataset, val_dataset, **train_kwargs):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
        train_args = self._get_train_args(**train_kwargs)
        trainer = transformers.Trainer(model=model,
                                       args=train_args,
                                       data_collator=data_collator,
                                       train_dataset=train_dataset,
                                       eval_dataset=val_dataset,
                                       )
        trainer.train()

    def _get_train_args(self, nepochs: int, eval_every: int, batch_size: int, save_every: int
                        ) -> transformers.TrainingArguments:
        training_arguments = transformers.TrainingArguments(
            output_dir=self.checkpoints_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=eval_every,
            save_steps=save_every,
            num_train_epochs=nepochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=3
        )
        return training_arguments

    def _get_last_save_name(self) -> str:
        checkpoint_dir_files = glob(os.path.join(self.checkpoints_dir, "*"))
        sorted_by_step_decrease = sorted(checkpoint_dir_files)[::-1]
        return sorted_by_step_decrease[0]


class TextsDataset(torch_data.Dataset):
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
