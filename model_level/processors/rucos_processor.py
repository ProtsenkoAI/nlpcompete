import transformers

from ..rucos_types import *

class RucosProcessor:
    # TODO: the class is too large, maybe add assistant components
    def __init__(self, mname: str):
        self.mname = mname
        self.maxlen = 512
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mname)

    def get_init_kwargs(self) -> dict:
        return {"mname": self.mname}

    def preprocess(self, features: UnprocFeatures, labels: UnprocLabels=None, device: torch.device=None
                   ) -> Union[ProcFeatures, Tuple[ProcFeatures, ProcLabels]]:
        tokenized = self.tokenize(*features)
        features_proc = self._preproc_tokenized(tokenized, device)
        if not labels is None:
            labels_proc = self._create_tensors([labels], device)[0].float()
            return features_proc, labels_proc
        return features_proc

    def postprocess(self, preds: ModelPreds, src_features: UnprocSubmFeatures) -> ProcSubmPreds:
        text1, text2, text_id, start, end, placeholder = src_features
        probs = preds.squeeze().cpu().detach().numpy()
        return text_id, probs, start, end, placeholder

    def tokenize(self, *texts):
        encoded = self.tokenizer(*texts,
                                 max_length=self.maxlen,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="np")
        return encoded

    def _preproc_tokenized(self, tokenized, device):
        needed_parts = [tokenized["input_ids"],
                        tokenized["attention_mask"],
                        tokenized["token_type_ids"]
                        ]
        tensors = self._create_tensors(needed_parts, device)
        assert len(tensors) == 3
        return tuple(tensors)

    def _create_tensors(self, arrays, device) -> List[torch.Tensor]:
        tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
            if not device is None:
                arr = arr.to(device, copy=True)
            tensors.append(arr)
        return tensors