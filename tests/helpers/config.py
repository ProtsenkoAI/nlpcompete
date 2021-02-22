import pathlib
import torch


class TestsConfig:
    def __init__(self):
        self.config_path = pathlib.Path(__file__)
        depth_from_root = 3
        self.project_root = self.config_path.parents[depth_from_root]
        self.data_dir = self.project_root / "data"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_path = str(self.data_dir / "sbersquad_train.json")
        self.test_path = str(self.data_dir / "sbersquad_test.json")

        self.model_name = "DeepPavlov/rubert-base-cased"

        self.batch_size = 8
