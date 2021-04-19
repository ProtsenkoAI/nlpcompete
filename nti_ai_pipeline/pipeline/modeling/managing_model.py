import torch

from .base_processor import BaseProcessor
from ..saving.model_and_processor_saver import ModelAndProcessorSaver
from .model_with_transformer import ModelWithTransformer

from typing import Union, Tuple, Optional
from .types import ModelPreds, ProcLabels


class ModelManager:
    # TODO: refactor problems with returning probs from postproc (and parameterising it)
    r"""
    Combines a torch model with a processor to provide functionality used by any other components.
    Model and processor shouldn't be used by themselves, without manager (except some torch-related
    operations with model, eg getting parameters of model to save or create optimizer).
    """
    def __init__(self, model: ModelWithTransformer, processor: BaseProcessor, device: torch.device,
                 saver: ModelAndProcessorSaver):
        self.model = model
        self.processor = processor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.saver = saver

    def get_model(self) -> ModelWithTransformer:
        return self.model

    def reset_model_weights(self):
        self.model.reset_weights()
        self.model.to(self.device)

    def preproc_forward(self, features,
                        labels=None) -> Union[ModelPreds, Tuple[ModelPreds, ProcLabels]]:
        r"""
        Preprocesses features and labels (if provided), passes obtained features to model, returns raw
        prediction tensors and processed labels (if provided)
        :param features:
        :param labels:
        """
        # TODO: operate with named tuples so we can preprocess features and labels, then put features to model using
        #   out.features, and return processed labels with out.labels

        out = self.processor.preprocess(features, labels=labels, device=self.device)
        preds_raw = self.model(out.features)
        preds = self.processor.after_forward(preds_raw)
        if labels is not None:
            return preds, out.labels
        return preds

    def predict_postproc(self, features, postproc_kwargs: Optional[dict] = None):
        r"""
        Takes features, returns postprocessed model predictions.
        """
        if postproc_kwargs is None:
            postproc_kwargs = {}
        # TODO: wtf? why features aren't batched (size=(8, ...))?
        text1, text2, question_idx, start, end, placeholder = features
        out = self.preproc_forward(features=(text1, text2, placeholder))
        preds = out
        postproc_preds = self.processor.postprocess(preds, features, **postproc_kwargs)
        return postproc_preds

    def save_model(self) -> str:
        name = self.saver.save(self.model, self.processor)
        return name

    def load_checkpoint(self, name: str):
        r"""
        Loads manager using checkpoint of saver (name). Name is returned by save_model() method
        """
        model, processor = self.saver.load(name)
        self.model = model
        self.processor = processor
