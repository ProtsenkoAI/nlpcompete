import torch


class ModelManager:
    # TODO: move train low-level operations to some class that'll be composed by modelmanager
    def __init__(self, model, processor, device):
        # TODO: add device support
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

    def predict_get_text(self, features, labels=None):
        out = self.preproc_forward(features, labels)
        if labels is None:
            preds = out
            postproc_preds = self.processor.postprocess(preds)
        else:
            preds, preproc_labels = out
            postproc_preds, postproc_labels = self.processor.postprocess(preds, preproc_labels)
            ground_truth_text = self.processor.text_from_token_idxs(postproc_labels, features)

        pred_answers = self.processor.text_from_token_idxs(postproc_preds, features)
        if not labels is None:
            return pred_answers, ground_truth_text
        return pred_answers

    def save_model(self, saver):
        return saver.save(self.model, self.processor)

    @classmethod
    def load(cls, saver, name, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = saver.load(name)
        return cls(model, processor, device)
