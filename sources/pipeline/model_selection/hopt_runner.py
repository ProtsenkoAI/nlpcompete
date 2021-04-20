from abc import ABC, abstractmethod
from pipeline.modeling import ModelManager
from pipeline.training import Trainer


class BaseHoptRunner(ABC):
    def __init__(self, cross_validator, hyper_opter,
                 data_container,
                 model_kwargs: dict, trainer_kwargs: dict):
        self.cv = cross_validator
        self.hopter = hyper_opter
        self.data_container = data_container
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs

    def _run(self, hopt_space: dict):
        self._validate_space_format(hopt_space)
        trainer_params = hopt_space["trainer_params"]
        model_params = hopt_space["model_params"]

        model = self.create_manager(self.model_kwargs, model_params)
        trainer = self.create_trainer(self.trainer_kwargs, trainer_params)

        final_score = self.cv.run(trainer, model, self.data_container)
        return -1 * final_score

    def _validate_space_format(self, space: dict):
        if "trainer_params" not in space or "model_params" not in space:
            raise ValueError("Space should contain trainer_params and model_params keys")

    @abstractmethod
    def create_manager(self, params_from_init: dict, params_from_hopt_space: dict) -> ModelManager:
        ...

    @abstractmethod
    def create_trainer(self, params_from_init: dict, params_from_hopt_space: dict) -> Trainer:
        ...
