# import unittest
# from collections import Collection
# from tests.helpers import config, std_objects
# config = config.TestsConfig()
#
#
# class SharedObjects:
#     def __init__(self):
#         self.manager = std_objects.get_model_manager()
#         self.trainer = std_objects.get_trainer()
#         self.data_assistant = std_objects.get_data_assistant()
#         self.dataset = std_objects.get_train_dataset(nrows=10)
#
#
# shared_objs = SharedObjects()
#
#
# class TestCrossValidator(unittest.TestCase):
#     def test_run(self):
#         cross_validator = std_objects.get_cross_validator()
#         res = cross_validator.run(shared_objs.trainer, shared_objs.manager, shared_objs.dataset,
#                                   shared_objs.data_assistant, fit_kwargs={"max_step": 2, "steps_betw_evals": 1},
#                                   nfolds=10, max_fold=1)
#         self.assertIsInstance(res, Collection)
