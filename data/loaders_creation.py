import torch.utils.data as torch_data


class DataLoaderSepPartsBuilder:
    # TODO: change logic of working with loaders in whole project
    def __init__(self, batch):
        self.batch = batch

    def build(self, dataset, has_answers=True):
        def sep_xy(objects_list):
            zipped_out = self._sep_parts(objects_list)
            if has_answers:
                features, labels = zipped_out
                features = self._sep_parts(features)
                labels = self._sep_parts(labels)
                return features, labels
            else:
                features = zipped_out
                return features

        return torch_data.DataLoader(dataset=dataset,
                                     collate_fn=sep_xy,
                                     batch_size=self.batch)

    def _sep_parts(self, samples_list):
        categories = list(zip(*samples_list))
        return categories