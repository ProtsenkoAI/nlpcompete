import torch.utils.data as torch_data
import torch

class DataLoaderSepXYCreator:
    # TODO: change logic of working with loaders in whole project
    # TODO: don't convert to tensor, move this functionality to models' managers
    def __init__(self, dataset, batch):
        self.dataset = dataset
        self.has_answers = self.dataset.has_answers
        self.batch = batch

    def get(self):
        return self._create_dataloader(self.dataset)

    def n_samples(self):
        return len(self.dataset)

    def get_subset(self, idxs):
        subset = torch_data.Subset(self.dataset, idxs)
        return self._create_dataloader(subset)

    def _create_dataloader(self, dataset): 
        def sep_xy(objects_list):
            zipped_out = list(zip(*objects_list))
            if self.has_answers:
                features, labels = zipped_out
                labels = self._zip_convert_to_tensor(labels)
            else:
                features = zipped_out
            
            features = self._zip_convert_to_tensor(features)

            if self.has_answers:
                return features, labels
            return features

        return torch_data.DataLoader(dataset=dataset,
                                     collate_fn=sep_xy,
                                     batch_size=self.batch)

    def _zip_convert_to_tensor(self, samples_list):
        categories = list(zip(*samples_list))
        tensors = []
        for category in categories:
            stacked = torch.stack(category)
            tensors.append(stacked)
        return tensors