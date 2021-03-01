from torch.utils import data as torch_data


class SizedDataset(torch_data.Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
