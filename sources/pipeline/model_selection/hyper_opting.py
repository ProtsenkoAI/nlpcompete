import os
import pickle
import hyperopt
from typing import Callable
from .types import ParamsWithResults


class HyperOpter:
    # TODO: the class violates single responsibility principle (storing trials, running opt, reversing metric value),
    #   have to separate to parts
    def __init__(self, func: Callable, higher_is_better=False, trials_save_dir="./",
                 trials_file_name="saved_trials"):
        self.func = self._wrap_if_needed(func, higher_is_better)
        self.trials_path = self._create_trials_path(trials_save_dir, trials_file_name)

    def _wrap_if_needed(self, func: Callable, higher_val_is_better: bool):
        if higher_val_is_better:
            def wrapped(*args, **kwargs):
                res = func(*args, **kwargs)
                return -1 * res

            return wrapped
        return func

    def _create_trials_path(self, dirr: str, filename: str):
        extension = ".json"
        return os.path.join(dirr, filename + extension)

    def optimize(self, space, ntrials: int):
        trials = self._load_trials()
        ntrials_already_done = len(trials.statuses())
        try:
            best_params = self._fmin_get_best_params(space, ntrials + ntrials_already_done, trials)
        except KeyError:
            # delete old trials and start with new
            trials = hyperopt.Trials()
            best_params = self._fmin_get_best_params(space, ntrials, trials)
        self._save_results(trials)
        return best_params

    def _fmin_get_best_params(self, space, ntrials: int, trials: hyperopt.Trials):
        return hyperopt.fmin(self.func,
                             space=space,
                             max_evals=ntrials,
                             trials=trials,
                             algo=hyperopt.tpe.suggest)

    def _load_trials(self) -> hyperopt.Trials:
        """
        # At the moment deletes old trials if they contain any unused params.
        # Maybe we can go more thrifty way
        """
        if os.path.isfile(self.trials_path):
            trials = pickle.load(open(self.trials_path, "rb"))
        else:
            trials = hyperopt.Trials()
        return trials

    def _save_results(self, trials: hyperopt.Trials):
        pickle.dump(trials, open(self.trials_path, "wb"))

    def get_params_and_results(self) -> ParamsWithResults:
        vals = []
        trials = self._load_trials()
        for trial in trials.trials:
            metric_val = trial["result"]["loss"]
            params = trial["misc"]["vals"]
            vals.append({"params": params, "metric": metric_val})
        return vals
