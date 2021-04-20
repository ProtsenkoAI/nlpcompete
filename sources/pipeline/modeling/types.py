from typing import NamedTuple, Any


BatchWithLabels = NamedTuple("BatchWithLabels", [("features", Any), ("labels", Any)])
BatchWithoutLabels = NamedTuple("BatchWithoutLabels", [("features", Any)])

ModelPreds = Any
ProcLabels = Any
