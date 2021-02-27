from torch.utils import data as torch_data

from ..types.dataset import SampleFeatures


class SizedDataset(torch_data.Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> SampleFeatures:
        raise NotImplementedError
