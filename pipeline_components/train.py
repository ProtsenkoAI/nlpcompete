import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from typing import Union

from model_level.evaluating import Validator
from model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
from model_level.saving.local_saver import LocalSaver
from model_level.managing_model import ModelManager


class Trainer:
    def __init__(self, validator: Validator, weights_updater:  QAWeightsUpdater, saver: LocalSaver):
        self.validator = validator
        self.weights_updater = weights_updater
        self.saver = saver

        self.eval_vals = []
        self.step_nb = 0
        self.epoch_nb = 0

        self.best_model_name = None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, model_manager: ModelManager,
            max_epoch:Union[None, int]=None, max_step:Union[None, int]=None,
            stop_patience:Union[None, int]=2, steps_betw_evals=200):

        model_manager.reset_model_weights()
        num_train_steps = self._calc_num_train_steps(max_epoch, max_step, len(train_loader))
        self.weights_updater.prepare_for_fit(model_manager, num_train_steps)

        losses = []
        while True:
            for batch in tqdm(train_loader):
                if self._early_stopping(max_epoch, max_step, stop_patience):
                    return
                loss_val = self.weights_updater.fit_with_batch(model_manager, batch)    
                losses.append(loss_val)

                if (self.step_nb + 1) % steps_betw_evals == 0:
                    self._eval_save_if_need(model_manager, val_loader)
                    print("Mean losses:", np.mean(losses))
                    losses = []
                if (self.step_nb + 1) % 200 == 0:
                    print('Mean losses:', np.mean(losses))
                    losses = []
                self.step_nb += 1
            self.epoch_nb += 1

    def _eval_save_if_need(self, manager, loader):
        manager.get_model().eval()
        eval_value = self.validator.eval(manager, loader)
        safe_max = 0        
        print("_eval. Eval_value:", eval_value)
        if len(self.eval_vals) > 0:
            safe_max = max(self.eval_vals)
        if eval_value >= safe_max:
            self.best_model_name = manager.save_model(self.saver)
        self.eval_vals.append(eval_value)
        manager.get_model().train()

    def load_best_manager(self) -> ModelManager:
        if self.best_model_name is None:
            raise ValueError("Model was not saved")
        return ModelManager.load(self.saver, self.best_model_name)

    def get_best_model_name(self):
        return self.best_model_name

    def _early_stopping(self, max_epoch: int, max_step: int, stop_patience: int) -> bool:
        if not max_step is None:
            if self.step_nb >= max_step:
                return True 
        if not max_epoch is None:
            if self.epoch_nb >= max_epoch:
                return True
        if self._exceeded_stopping_patience(stop_patience):
            return True
            
        return False

    def _exceeded_stopping_patience(self, stop_patience) -> bool:
        steps_passed = len(self.eval_vals)
        enough_steps_passed = steps_passed > stop_patience
        if enough_steps_passed:
            best_result_step = np.argmax(self.eval_vals)
            if (steps_passed - best_result_step - 1) >= stop_patience:
                print("EARLY STOPPING", "eval_vals", self.eval_vals, "best_result_step", best_result_step)
                return True
        return False

    def _calc_num_train_steps(self, max_epoch, max_step, steps_in_epoch) -> int:
        if max_step is None and max_epoch is None:
            raise ValueError
        elif max_step is None:
            return max_epoch * steps_in_epoch
        else:
            return max_step

    def get_eval_vals(self) -> list:
        return self.eval_vals
