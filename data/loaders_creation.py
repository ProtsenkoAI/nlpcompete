import torch.utils.data as torch_data
import torch

class DataLoaderSepPartsBuilder:
    # TODO: change logic of working with loaders in whole project
    def __init__(self, batch):
        self.batch = batch

    def build(self, dataset):
        def sep_xy(objects_list):
            zipped_out = list(zip(*objects_list))
            if dataset.has_answers:
                features, labels = zipped_out
                labels = self._sep_parts(labels)
            else:
                features = zipped_out
            
            features = self._sep_parts(features)

            if dataset.has_answers:
                return features, labels
            return features

        return torch_data.DataLoader(dataset=dataset,
                                     collate_fn=sep_xy,
                                     batch_size=self.batch)

    def _sep_parts(self, samples_list):
        categories = list(zip(*samples_list))
        return categories