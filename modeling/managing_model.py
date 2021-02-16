import os
import random
import torch
from .transformer_qanda import TransformerQA


class ModelManager:
    save_dir = "./saved_models/"
    os.makedirs(save_dir, exist_ok=True)

    def __init__(self, model, processor):
        # TODO: add device support
        self.model = model
        self.processor = processor
    def preproc_forward(self, features):
        proc_feats = self.processor.preprocess_features(features)
        preds = self.model(proc_feats)
        return preds

    def preproc_labels(self, labels):
        proc_labels = self.processor.preprocess_labels(labels)
        return proc_labels

    def reset_model_weights(self):
        self.model.reset_weights()

    def predict_postproc(self, features):
        preds = self.preproc_forward(features)
        processed_out = self.processor.postprocess_preds(preds)
        return processed_out

    # TODO: maybe add saver to save/load models. They can be local, support BD etc.
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
