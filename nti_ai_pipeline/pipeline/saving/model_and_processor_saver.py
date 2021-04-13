from abc import ABC, abstractmethod
from torch import nn
from ..modeling.base_processor import BaseProcessor
from typing import Tuple, Optional, Type
from .. import util


class ModelAndProcessorSaver(ABC):
    @abstractmethod
    def save(self, model: nn.Module, processor: BaseProcessor) -> str:
        ...

    @abstractmethod
    def load(self, name: str,
             model_class: Optional[Type[nn.Module]] = None,
             processor_class: Optional[Type[BaseProcessor]] = None) -> Tuple[nn.Module, BaseProcessor]:
        r"""
        If mode class and processor_class are not provided will try to take the classes memorized from previous save.
        If save() wasn't called after initialization, raises error
        """
        ...

    def get_tqdm(self, mode: str):
        return util.get_tqdm_obj(mode)
