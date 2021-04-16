from abc import ABC, abstractmethod
from typing import Optional


class PseudoLabeler(ABC):
    # TODO
    def __init__(self):
        ...

    def run(self, model_manager, data_container, threshold: Optional[float] = None,
            samples_num: Optional[int] = None,):
        # TODO
        ...
