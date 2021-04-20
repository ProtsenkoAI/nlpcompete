from typing import Tuple
from torch.utils.data import DataLoader

from .base_cross_validator import BaseCrossValidator
from pipeline.data.types import Samples


class CrossValidator(BaseCrossValidator):
    def __init__(self, train_dataset_class, val_dataset_class, batch_size=8,):
        self.batch_size = batch_size
        self.train_ds_cls = train_dataset_class
        self.val_ds_cls = val_dataset_class
        super().__init__()

    def create_loaders(self, train_samples: Samples, val_samples: Samples) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(self.train_ds_cls(train_samples), batch_size=self.batch_size)
        val_loader = DataLoader(self.val_ds_cls(val_samples), batch_size=self.batch_size)
        return train_loader, val_loader
