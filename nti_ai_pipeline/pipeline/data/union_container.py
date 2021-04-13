from .base_container import BaseContainer
from .types import Samples


class UnionContainer(BaseContainer):
    def __init__(self, *containers: BaseContainer):
        self.containers = containers
        super().__init__()

    def collect_data(self) -> Samples:
        data = []
        for container in self.containers:
            data += container.collect_data()
        return data
