import numpy as np
from tqdm.notebook import tqdm


class Trainer:
    # TODO: manage answers out of scope (512 tokenss)
    def __init__(self, validator, loader_builder, weights_updater):
        self.loader_builder = loader_builder
        self.validator = validator
        self.weights_updater = weights_updater

        self.eval_vals = []
        self.step_nb = 0
        self.epoch_nb = 0

    def fit(self, train_dataset, test_dataset, model_manager, max_epoch=None, max_step=None, 
            stop_patience=2, steps_betw_evals=200):
        # TODO: get one dataset and split it inside this class
        # TODO: reset weights when starting training
        train_loader = self.loader_builder.build(train_dataset, has_answers=True)
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
                    self._eval(model_manager, test_dataset)
                    print("Mean losses:", np.mean(losses))
                    losses = []
                
                self.step_nb += 1
            self.epoch_nb += 1

    def _eval(self, manager, dataset):
        manager.get_model().eval()
        eval_value = self.validator.eval(manager, dataset)
        self.eval_vals.append(eval_value)
        manager.get_model().train()

    def _early_stopping(self, max_epoch, max_step, stop_patience):
        if not max_step is None:
            if self.step_nb >= max_step:
                return True 
        if not max_epoch is None:
            if self.epoch_nb >= max_epoch:
                return True
        if self._exceeded_stopping_patience(stop_patience):
            return True
            
        return False

    def _exceeded_stopping_patience(self, stop_patience):
        steps_passed = len(self.eval_vals)
        enough_steps_passed = steps_passed > stop_patience
        if enough_steps_passed:
            best_result_step = np.argmax(self.eval_vals)
            if (steps_passed - best_result_step - 1) >= stop_patience:
                print("EARLY STOPPING", "eval_vals", self.eval_vals, "best_result_step", best_result_step)
                return True

    def _calc_num_train_steps(self, max_epoch, max_step, steps_in_epoch):
        if max_step is None and max_epoch is None:
            raise ValueError
        elif max_step is None:
            return max_epoch * steps_in_epoch
        else:
            return max_step

    def get_eval_vals(self):
        return self.eval_vals
