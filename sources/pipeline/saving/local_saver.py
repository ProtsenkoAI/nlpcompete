from typing import Dict, Tuple, Type, Optional
from torch import nn

from .model_and_processor_saver import ModelAndProcessorSaver
from .static_local_saver import StaticLocalSaver
from ..modeling import BaseProcessor


class LocalSaver(ModelAndProcessorSaver):
    """Uses StaticLocalSaver under the hood, but has memory for saved classes: if you saved model and processor,
    and then try to load them, you don't need to provide their classes for init: the classes were saved before.
    """
    def __init__(self, *args, **kwargs):
        self._static_saver = StaticLocalSaver(*args, **kwargs)
        self.name_to_classes: Dict[str, Tuple[Type[nn.Module], Type[BaseProcessor]]] = {}

    def save(self, model: nn.Module, processor: BaseProcessor) -> str:
        name = self._static_saver.save(model, processor)
        self.name_to_classes[name] = model.__class__, processor.__class__
        return name

    def load(self, name: str,
             model_class: Optional[Type[nn.Module]] = None,
             processor_class: Optional[Type[BaseProcessor]] = None) -> Tuple[nn.Module, BaseProcessor]:
        if model_class is None:
            self._check_name_exists(name)
            model_class = self.name_to_classes[name][0]
        if processor_class is None:
            self._check_name_exists(name)
            processor_class = self.name_to_classes[name][1]
        return self._static_saver.load(name, model_class, processor_class)

    def _check_name_exists(self, name: str):
        if name not in self.name_to_classes:
            raise RuntimeError("Doesn't provide model or/and processor class,"
                               "but the model was not saved by this saver, so can't get them from memory")
