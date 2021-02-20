import os
import random
import torch
from .transformer_qanda import TransformerQA


class ModelManager:
    # TODO: move train low-level operations to some class that'll be composed by modelmanager
    save_dir = "./saved_models/"
    os.makedirs(save_dir, exist_ok=True)

    def __init__(self, model, processor, device=None):
        # TODO: add device support
        self.model = model
        self.processor = processor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def preproc_forward_labeled(self, features, labels):
        proc_feats, proc_labels = self.processor.preprocess_features_and_labels(features, labels, self.device)
        preds = self.model(proc_feats)
        return preds, proc_labels
        
    def preproc_forward(self, features):
        proc_feats = self.processor.preprocess_features(features, self.device)
        preds = self.model(proc_feats)
        return preds

    # def preproc_labels(self, labels, offsets_mapping):
    #     proc_labels = self.processor.preprocess_labels(labels, self.device)
    #     return proc_labels

    def predict_postproc(self, features):
        preds = self.preproc_forward(features)
        processed_out = self.processor.postprocess_preds(preds)
        answers = self.processor.text_from_preds(processed_out, features)
        return processed_out

    def predict_postproc_labeled(self, features, labels):
        preds, labels_proc = self.preproc_forward_labeled(features, labels)
        process_preds = self.processor.postprocess_preds(preds)
        non_tensor_labels = self.processor.postprocess_labels(labels_proc)
        labels_start_end = list(zip(*non_tensor_labels))
        return process_preds, labels_start_end

    def save_model(self):
        generated_name = f"lol_model{random.randint(0, 1000000)}_weights.pt"
        path2model_weights = os.path.join(self.save_dir, generated_name)
        torch.save(self.model.state_dict(), path2model_weights)
        return generated_name

    @classmethod
    def load_model(cls, name):
        path2model_weights = os.path.join(cls.save_dir, name)
        model = TransformerQA("DeepPavlov/rubert-base-cased")
        saved_state = torch.load(path2model_weights)
        model.load_state_dict(saved_state)
        return model
