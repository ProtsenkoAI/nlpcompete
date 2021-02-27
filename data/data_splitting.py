from torch.utils import data as torch_data
from sklearn.model_selection import train_test_split


class DataAssistant:
    def __init__(self, loader_builder):
        self.loader_builder = loader_builder

    def train_test_split(self, dataset: torch_data.Dataset, test_size=0.2):
        dataset_indexes = list(range(len(dataset)))
        train_idxs, test_idxs = train_test_split(dataset_indexes, test_size=test_size)

        train_dataset = torch_data.Subset(dataset, train_idxs)
        val_dataset = torch_data.Subset(dataset, test_idxs)
        train_loader = self.loader_builder.build(train_dataset, has_answers=True)
        test_loader = self.loader_builder.build(val_dataset)

        return train_loader, test_loader

