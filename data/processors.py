import transformers
import torch


class QADataProcessor:
    # TODO: maybe move converting from batches to categories from loaders to processor
    def __init__(self, mname):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)
        self.maxlen = 512

    def preprocess_features(self, features):
        contexts, questions = features
        proc_features = self._preproc_texts(contexts, questions)
        return proc_features

    def preprocess_labels(self, start_and_end_idxs):
        labels_proc = []
        for labels_part in start_and_end_idxs:
            part_proc = torch.tensor(labels_part)
            labels_proc.append(part_proc)
        return labels_proc

    def postprocess_preds(self, preds):
        start_logits, end_logits = preds
        start_argmax = torch.argmax(start_logits)
        end_argmax = torch.argmax(end_logits)
        return int(start_argmax), int(end_argmax)


    def _preproc_texts(self, *texts):
        encoded = self.tokenizer(*texts,
                                max_length=self.maxlen,
                                padding="max_length",
                                truncation="longest_first",
                                return_tensors="pt")

        encoded_proc = self._proc_tokenizer_out(encoded)
        return encoded_proc

    def _proc_tokenizer_out(self, tokenizer_out):
        # maybe later we'll need a tokenizer wrapper to return this parts
        needed_parts = [tokenizer_out["input_ids"], 
                        tokenizer_out["attention_mask"], 
                        tokenizer_out["token_type_ids"]
                        ]
        return needed_parts