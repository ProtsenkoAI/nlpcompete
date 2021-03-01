import torch
from torch import nn
from typing import Tuple, List, Union

from model_level.processors.rucos_processor import RucosProcessor
from model_level.saving.local_saver import LocalSaver

from .rucos_types import *

class ModelManager:
    def __init__(self, model: nn.Module, processor: RucosProcessor, device: torch.device):
        self.model = model
        self.processor = processor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def reset_processor(self, new_processor):
        self.processor = new_processor

    def reset_model_weights(self):
        self.model.reset_weights()
        self.model.to(self.device)

    def preproc_forward(self, features: Union[UnprocFeatures, UnprocSubmFeatures],
                        labels: UnprocLabels=None, mode="train") -> Union[ModelPreds,
                                                                                   Tuple[ModelPreds, ProcLabels]]:
        if mode == "subm":
            features = features[:2] # text and text2
        elif mode == "train":
            pass
        else:
            raise ValueError
        out = self.processor.preprocess(features, labels=labels, device=self.device)
        if labels is None:
            preds = self.model(out).squeeze()
            return preds
        else:
            proc_feats, labels_proc = out
            preds = self.model(proc_feats).squeeze(dim=-1)
            return preds, labels_proc

    def predict_postproc(self, features: UnprocSubmFeatures, src_labels: UnprocLabels=None) -> Union[Tuple[str],
                                                                                                 Tuple[Tuple[str], Tuple[str]]]:
        # TODO: bruh
        # if not src_labels is None:
        #     raise ValueError("can't pass labels in predict_postproc: we use it only in submission")
        out = self.preproc_forward(features, src_labels, mode="subm")
        if not src_labels is None:
            preds, proc_labels = out
            postproc_preds = self.processor.postprocess(preds, features)
            return postprocess_preds, proc_labels
        preds = out
        postproc_preds = self.processor.postprocess(preds, features)
        return postproc_preds

    def save_model(self, saver: LocalSaver) -> str:
        return saver.save(self.model, self.processor)

    @classmethod
    def load(cls, saver: LocalSaver, name: str, device:Union[None, torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = saver.load(name)
        return cls(model, processor, device)
