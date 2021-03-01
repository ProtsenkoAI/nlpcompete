from typing import Iterable, Tuple

import torch


class UnbatchingProcessor:
    """
    Splits batched output from model
    eg.:
    (
        tensor(
            [1, 2, 3],
            [4, 5, 6]
        ),
        tensor(
            [7, 8, 9],
            [10, 11, 12]
        )
    )
    ->
    [
        (tensor([1, 2, 3]), tensor([7, 8, 9])),
        (tensor([4, 5, 6]), tensor([10, 11, 12]))
    ]
    ! This is iterable, not list
    """

    def preprocess(self, features: Tuple[torch.Tensor, torch.Tensor]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        tmp = torch.split(features[0], 1)
        # print(tmp)
        start_preds_reshaped = map(torch.squeeze, tmp)
        end_preds_reshaped = map(torch.squeeze, torch.split(features[1], 1))
        return list(zip(start_preds_reshaped, end_preds_reshaped))
