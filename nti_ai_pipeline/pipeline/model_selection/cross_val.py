from typing import Optional, Tuple
from ..data import BaseContainer

from abc import abstractmethod
from torch.utils.data import DataLoader
from ..training import Trainer
from ..modeling import ModelManager
from ..data.types import Samples


class CrossValidator:
    def run(self, trainer: Trainer, manager: ModelManager, container: BaseContainer, fit_kwargs: Optional[dict] = None,
            nfolds=5, max_fold=3) -> list:
        folds_results = []
        for train_samples, val_samples in container.kfold(nfolds=nfolds, used_folds=max_fold):
            train_loader, val_loader = self.create_loaders(train_samples, val_samples)
            trainer.fit(train_loader, val_loader, manager, **fit_kwargs)
            eval_vals = trainer.get_eval_vals()
            best_eval_val = max(eval_vals)
            folds_results.append(best_eval_val)
        return folds_results

    @abstractmethod
    def create_loaders(self, train_samples: Samples, val_samples: Samples) -> Tuple[DataLoader, DataLoader]:
        ...
