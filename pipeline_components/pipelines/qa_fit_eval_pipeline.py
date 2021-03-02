# from collections import Collection
# from model_level.saving.local_saver import LocalSaver
# from model_level.models.transformer_qanda import TransformerQA
# from model_level.processors import QADataProcessor
# from model_level.evaluating import Validator
# from model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
# from model_level.managing_model import ModelManager
#
# from pipeline_components.train import Trainer
#
#
# class QATrainEvalPipeline:
#     """Intended to define unchanging task-related pipeline values once and then
#     do all operations providing needed parameters (esp. model and training architecture).
#     Also supports standard kwargs passed to objects that can be redifined in run().
#     """
#     def __init__(self,  transformer_mname, models_save_dir,
#                  device, trainer_fit_kwargs={},
#                  weights_updater_standard_kwargs={},
#                  model_standard_kwargs={},
#                  processor_standard_kwargs={},
#                  ):
#         """Creates unchanging objects and saved params"""
#         # train_container = QADataContainer(path=train_path, nrows=nrows)
#         # self.train_dataset = StandardDataset(train_container)
#         # loader_builder = DataLoaderSepPartsBuilder(batch=batch_size)
#         # self.data_assistant = DataAssistant(loader_builder)
#
#         self.validator = Validator()
#         self.saver = LocalSaver(models_save_dir)
#
#         self.trainer_fit_kwargs = trainer_fit_kwargs
#         self.mname = transformer_mname
#         self.device = device
#         self.std_model_kwargs = model_standard_kwargs
#         self.std_processor_kwargs = processor_standard_kwargs
#         self.std_weights_kwargs = weights_updater_standard_kwargs
#
#     def run(self, train_loader, val_loader, weights_updater_kwargs=None, model_kwargs=None, processor_kwargs=None,
#                                     train_test_split_kwargs=None) -> list:
#
#         weights_updater_kwargs, model_kwargs, processor_kwargs, train_test_split_kwargs = \
#             self._set_to_dict_if_none(*weights_updater_kwargs, model_kwargs, processor_kwargs, train_test_split_kwargs)
#         # train_loader, val_loader = self.data_assistant.train_test_split(self.train_dataset, **train_test_split_kwargs)
#         manager = self._create_manager(model_kwargs, processor_kwargs)
#         trainer = self._create_trainer(weights_updater_kwargs)
#         trainer.fit(train_loader, val_loader, manager, **self.trainer_fit_kwargs)
#         val_scores = trainer.get_eval_vals()
#         return val_scores
#
#     def _set_to_dict_if_none(self, *args) -> Collection:
#         returned = []
#         for arg in args:
#             if arg is None:
#                 arg = {}
#             returned.append(arg)
#         return returned
#
#     def _create_manager(self, run_model_kwargs, run_proc_kwargs) -> ModelManager:
#         model_kwargs = self.std_model_kwargs.copy()
#         model_kwargs.update(run_model_kwargs)
#         proc_kwargs = self.std_processor_kwargs.copy()
#         proc_kwargs.update(run_proc_kwargs)
#
#         model = TransformerQA(self.mname, **model_kwargs)
#         proc = QADataProcessor(self.mname, **proc_kwargs)
#
#         return ModelManager(model, proc, device=self.device)
#
#     def _create_trainer(self, run_weights_updater_kwargs) -> Trainer:
#         weights_update_kwargs = self.std_weights_kwargs.copy()
#         weights_update_kwargs.update(run_weights_updater_kwargs)
#
#         weights_updater = QAWeightsUpdater(**weights_update_kwargs)
#         return Trainer(self.validator, weights_updater, self.saver)
