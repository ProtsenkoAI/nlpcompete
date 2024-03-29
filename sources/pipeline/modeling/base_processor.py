from abc import ABC, abstractmethod
from typing import Optional, Union
from .types import BatchWithLabels, BatchWithoutLabels
import torch


class BaseProcessor(ABC):
    # TODO: in both QAProcAssistant and RucosProcessor there are functions to convert between tokens and char idxs
    #   so it will be cool to create util methods for this purpose
    @abstractmethod
    def preprocess(self, features, labels=None, device: Optional[torch.device] = None
                   ) -> Union[BatchWithLabels, BatchWithoutLabels]:
        ...

    @abstractmethod
    def after_forward(self, raw_model_out):
        return raw_model_out.squeeze(dim=-1)  # TODO: move to subtypes

    @abstractmethod
    def prepare_to_preproc_forward(self, raw_features):
        # TODO: eliminate this method.
        #  it's bad practice: trainer/DataPredictor should prepare data from raw type (like SubmSample) to type
        #   appropriate for processor and manager
        ...

    @abstractmethod
    def postprocess(self, model_out, features, **kwargs):
        ...

    @abstractmethod
    def get_init_kwargs(self) -> dict:
        """Returns everything that need to be passed in __init__ when saver will load the processor"""
        ...
