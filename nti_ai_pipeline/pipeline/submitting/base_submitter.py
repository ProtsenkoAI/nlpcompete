from pipeline.modeling import ModelManager
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
from typing import Any, List
import os
import pickle
from .. import util


class Submitter(ABC):
    # TODO: have to use data_predictor here (!)
    SubmObj = Any
    SamplePrediction = Any

    def __init__(self, subm_dir: str, has_to_save_preds=False, tqdm_mode="notebook"):
        self.subm_dir = subm_dir
        self._has_to_save_preds = has_to_save_preds
        self.tqdm_mode = tqdm_mode

    def create_submission(self, model_manager: ModelManager, data_loader, subm_file_name: str, preds_file_name=None):
        preds = self.get_test_predictions(model_manager, data_loader)
        if self._has_to_save_preds:
            preds_path = os.path.join(self.subm_dir, preds_file_name)
            self.save_preds(preds, preds_path)
        subm_obj = self.form_submission(preds)

        sub_path = os.path.join(self.subm_dir, subm_file_name)
        self.write_submission(subm_obj, sub_path)

    def save_preds(self, preds: List[SamplePrediction], path: str):
        with open(path, "wb") as f:
            pickle.dump(preds, f)

    def get_test_predictions(self, model_manager: ModelManager, loader: DataLoader) -> List[SamplePrediction]:
        test_preds = []
        for batch in self.get_tqdm_obj()(loader):
            preds = self.pred_batch(batch, model_manager)
            test_preds += preds
        return test_preds

    @abstractmethod
    def pred_batch(self, batch, model_manager) -> List[SamplePrediction]:
        ...

    @abstractmethod
    def form_submission(self, preds: List[SamplePrediction]) -> SubmObj:
        ...

    @abstractmethod
    def write_submission(self, subm_obj: SubmObj, sub_path: str):
        ...

    def get_tqdm_obj(self):
        return util.get_tqdm_obj(self.tqdm_mode)
