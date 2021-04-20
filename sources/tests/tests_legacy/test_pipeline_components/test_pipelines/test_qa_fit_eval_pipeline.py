import unittest

from sources.pipeline import QATrainEvalPipeline

from tests.helpers import config

config = config.TestsConfig()


class TestHyperOpter(unittest.TestCase):
    def test_run(self):
        print("test_run pipeline")
        train_file_path = config.train_path
        # setting lr to str to check that later pipeline overrides it with lr passed to run()
        pipeline = QATrainEvalPipeline(train_file_path, config.model_name, config.save_dir,
                                       config.device, config.batch_size,
                                       trainer_fit_kwargs={"max_step": 2, "steps_betw_evals": 1},
                                       weights_updater_standard_kwargs={"lr": "unvalid_param",
                                                                        "warmup": 350},
                                       model_standard_kwargs={"head_nlayers": "unvalid_param"},
                                       nrows=2)
        eval_vals = pipeline.run(weights_updater_kwargs={"lr": 1e-5},
                                 model_kwargs={"head_nlayers": 1},
                                 processor_kwargs={},
                                 train_test_split_kwargs={"test_size": 0.5})
        self.assertEqual(len(eval_vals), 2)
