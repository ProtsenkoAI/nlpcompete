import unittest
from collections import Collection
from pipeline_components.pipelines.qa_pipeline import QAPipeline

from model_level.managing_model import ModelManager

from tests.helpers import config
config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        train_file_path = config.train_path
        self.pipeline = QAPipeline(train_file_path, config.model_name, config.save_dir,
                              config.device, config.batch_size,
                              trainer_fit_kwargs={"max_step": 2, "steps_betw_evals": 1},
                              weights_updater_standard_kwargs={"lr": "unvalid_param",
                                                                        "warmup": 350},
                              model_standard_kwargs={"head_nlayers": "unvalid_param"},
                              nrows=2, test_path=config.test_path)

        self.basic_run_kwargs = dict(weights_updater_kwargs={"lr": 1e-5},
                                     model_kwargs={"head_nlayers": 1},)


shared_objs = SharedObjects()


class TestHyperOpter(unittest.TestCase):
    def test_train_eval(self):
        print("test_run pipeline")
        # setting lr to str to check that later pipeline overrides it with lr passed to run()
        eval_vals = shared_objs.pipeline.train_get_eval_vals(**shared_objs.basic_run_kwargs,
                                                             train_test_split_kwargs={"test_size": 0.5})
        self.assertEqual(len(eval_vals), 2)

    def test_return_manager(self):
        print("test_return_manager")
        manager = shared_objs.pipeline.train_return_manager(**shared_objs.basic_run_kwargs,
                                                             train_test_split_kwargs={"test_size": 0.5})
        self.assertIsInstance(manager, ModelManager)

    def test_cross_val(self):
        print("test_cross_val")
        cross_val_value = shared_objs.pipeline.cross_val(**shared_objs.basic_run_kwargs,
                                                         cross_val_kwargs={"nfolds": 8, "max_fold": 1})
        self.assertIsInstance(cross_val_value, Collection)

    def test_submit(self):
        shared_objs.pipeline.fit_submit(**shared_objs.basic_run_kwargs,
                                       submitter_create_submission_kwargs={"subm_file_name": "some_subm"}, test_nrows=2)
