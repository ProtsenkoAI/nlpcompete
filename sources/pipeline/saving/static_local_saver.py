import torch
from torch import nn
import json
import os
import uuid
from typing import Tuple, Union, Type

from ..modeling import BaseProcessor, ModelWithTransformer
from .model_and_processor_saver import ModelAndProcessorSaver


class StaticLocalSaver(ModelAndProcessorSaver):
    r"""Saver that saves model and processor, and doesn't save any information about model and processor classes
    for later initialization in load()."""

    def __init__(self, save_dir="./saved_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model: nn.Module, processor: Union[BaseProcessor]) -> str:
        model_init_kwargs = model.get_init_kwargs()
        processor_init_kwargs = processor.get_init_kwargs()
        state_dict = model.state_dict()
        meta = {"model_meta": model_init_kwargs,
                "processor_meta": processor_init_kwargs}
        name = self._save_meta_and_state_dict(meta, state_dict)
        return name

    def _save_meta_and_state_dict(self, meta: dict, state: dict) -> str:
        name = self._create_model_name()
        meta_path = self._create_meta_path(name)
        weights_path = self._create_weights_path(name)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        torch.save(state, weights_path)
        return name

    def load(self, name: str,
             model_class: Type[ModelWithTransformer] = None,
             processor_class: Type[BaseProcessor] = None
             ) -> Tuple[ModelWithTransformer, BaseProcessor]:

        meta, state_dict = self._load_meta_and_state(name)
        model = model_class(**meta["model_meta"])
        model.load_state_dict(state_dict)
        processor = processor_class(**meta["processor_meta"])
        return model, processor

    def _load_meta_and_state(self, name):
        meta_path = self._create_meta_path(name)
        weights_path = self._create_weights_path(name)

        with open(meta_path) as f:
            meta = json.load(f)
        weights = torch.load(weights_path)
        return meta, weights

    def _create_model_name(self) -> str:
        model_name = str(uuid.uuid4())
        # while model_name in existing_models:
        meta_path = self._create_meta_path(model_name)
        while os.path.isfile(meta_path):  # model with name already exists so we create new name
            model_name = str(uuid.uuid4())
            meta_path = self._create_meta_path(model_name)
        return model_name

    def _create_meta_path(self, name: str) -> str:
        return os.path.join(self.save_dir, name + "_meta.json")

    def _create_weights_path(self, name: str) -> str:
        return os.path.join(self.save_dir, name + "weights.pt")
