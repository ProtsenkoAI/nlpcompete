from typing import List, NamedTuple

ParamsWithResults = List[NamedTuple("ParamsWithResults", [("params", dict), ("metric", float)])]
