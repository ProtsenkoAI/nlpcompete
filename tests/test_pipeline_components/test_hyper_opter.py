import unittest
from hyperopt import hp
from math import log

from pipeline_components.hyper_opting import HyperOpter
from tests.helpers import config, std_objects
config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.pipeline = std_objects.get_fit_eval_pipeline()
        weights_updater_space = {"lr": hp.loguniform("lr", log(1e-7), log(1e-2)),
                                 }
        model_space = {"head_nneurons": hp.quniform("head_nneurons", 50, 1000, q=50)
                       }

        self.space = {"weights_updater_kwargs": weights_updater_space,
                      "model_kwargs": model_space}


shared_objs = SharedObjects()


class TestHyperOpter(unittest.TestCase):
    def test_run_optimize(self):

        def run_pipeline_get_best_res(hopt_chosen_params):
            weights_kwargs = hopt_chosen_params["weights_updater_kwargs"]
            model_kwargs = hopt_chosen_params["model_kwargs"]
            eval_vals = shared_objs.pipeline.run(weights_updater_kwargs=weights_kwargs,
                                                 model_kwargs=model_kwargs)
            return max(eval_vals)

        hyper_opter = HyperOpter(func=run_pipeline_get_best_res,
                                 higher_is_better=True,
                                 trials_save_dir=config.save_dir,
                                 trials_file_name="saved_trials",
                                 )

        best_params = hyper_opter.optimize(space=shared_objs.space, ntrials=1)
        par_results = hyper_opter.get_params_and_results()
        score = par_results[0]["metric"]
        params = par_results[0]["params"]
        is_results_correct = isinstance(score, float) and "lr" in params and "head_nneurons" in params
        self.assertGreaterEqual(len(par_results), 1)
        self.assertTrue("params" in par_results[0] and "metric" in par_results[0])
        self.assertTrue(is_results_correct)

    def test_call_multiple_times_check_results_are_saved(self):
        def dumb_func(some_hopt_space):
            return 0
        hyper_opter = HyperOpter(func=dumb_func,
                                 trials_save_dir=config.save_dir,
                                 trials_file_name="saved_dumb_trials",
                                 )
        for i in range(3):
            hyper_opter.optimize(space=shared_objs.space, ntrials=2)

        saved_results = hyper_opter.get_params_and_results()
        self.assertGreaterEqual(len(saved_results), 3 * 2)
