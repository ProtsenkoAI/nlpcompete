import torch
from time import time


class ModelManager:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def reset_model_weights(self):
        self.model.reset_weights()
        self.model.to(self.device)

    def preproc_forward(self, features, labels=None):
        out = self.processor.preprocess(features, labels=labels, device=self.device)
        if labels is None:
            preds = self.model(out)
            return preds
        else:
            proc_feats, labels_proc = out
            preds = self.model(proc_feats)
            return preds, labels_proc

    def predict_postproc(self, features, labels=None):
        out = self.preproc_forward(features, labels)
        if labels is None:
            preds = out
            postproc_preds = self.processor.postprocess(preds, features)
        else:
            preds, _ = out
            postproc_preds, postproc_labels = self.processor.postprocess(preds, features, labels)
            return postproc_preds, postproc_labels
        return postproc_preds

    def save_model(self, saver):
        return saver.save(self.model, self.processor)

    @classmethod
    def load(cls, saver, name, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = saver.load(name)
        return cls(model, processor, device)
