from collections import Collection
from typing import Union, Iterable

from data.datasets.subm_dataset import SubmDataset

from model_level.saving.local_saver import LocalSaver
from model_level.models.sent_pair_binary_classifier import SentPairBinaryClassifier
from model_level.processors import RucosProcessor
from model_level.evaluating import Validator
from model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
from model_level.managing_model import ModelManager

from pipeline_components.train import Trainer
from pipeline_components.submitting import Submitter
from pipeline_components.cross_val import CrossValidator


class QAPipeline:
    """Intended to define unchanging task-related pipeline values once and then
    do all operations providing needed parameters (esp. model and training architecture).
    Also supports standard kwargs passed to objects that can be redifined in run().
    """
    def __init__(self, transformer_mname, models_save_dir,
                 device, trainer_fit_kwargs={},
                 weights_updater_standard_kwargs={},
                 model_standard_kwargs={},
                 processor_standard_kwargs={},
                 submitter_init_kwargs={},
                 cross_val_init_kwargs={}):
        """Creates unchanging objects and saved params"""
        self.submitter_init_kwargs = submitter_init_kwargs
        self.cross_val_init_kwargs = cross_val_init_kwargs

        self.validator = Validator()
        self.saver = LocalSaver(models_save_dir)

        self.trainer_fit_kwargs = trainer_fit_kwargs
        self.mname = transformer_mname
        self.device = device
        self.std_model_kwargs = model_standard_kwargs
        self.std_processor_kwargs = processor_standard_kwargs
        self.std_weights_kwargs = weights_updater_standard_kwargs

    def train_get_eval_vals(self, train_loader, val_loader, weights_updater_kwargs=None, model_kwargs=None, processor_kwargs=None
                            ) -> list:
        weights_updater_kwargs, model_kwargs, processor_kwargs = \
            self._set_to_dict_if_none(weights_updater_kwargs, model_kwargs, processor_kwargs)
        manager = self._create_manager(model_kwargs, processor_kwargs)
        trainer = self._create_trainer(weights_updater_kwargs)
        trainer.fit(train_loader, val_loader, manager, **self.trainer_fit_kwargs)
        return trainer.get_eval_vals()

    # def train_return_manager(self, weights_updater_kwargs=None, model_kwargs=None, processor_kwargs=None,
    #                                 train_test_split_kwargs=None) -> ModelManager:
    #     weights_updater_kwargs, model_kwargs, processor_kwargs, train_test_split_kwargs = \
    #         self._set_to_dict_if_none(weights_updater_kwargs, model_kwargs, processor_kwargs, train_test_split_kwargs)
    #
    #     trainer = self._create_trainer(weights_updater_kwargs)
    #     manager = self._create_manager(model_kwargs, processor_kwargs)
    #     self._fit(trainer, manager, train_test_split_kwargs)
    #     return manager


    # def fit_submit(self, weights_updater_kwargs=None, model_kwargs=None, processor_kwargs=None,
    #                train_test_split_kwargs=None, submitter_create_submission_kwargs=None, test_nrows=None):
    #     (weights_updater_kwargs, model_kwargs, processor_kwargs, train_test_split_kwargs,
    #      submitter_create_submission_kwargs) = (self._set_to_dict_if_none(weights_updater_kwargs, model_kwargs,
    #                                     processor_kwargs, train_test_split_kwargs, submitter_create_submission_kwargs))
    #
    #     trainer = self._create_trainer(weights_updater_kwargs)
    #     manager = self._create_manager(model_kwargs, processor_kwargs)
    #     self._fit(trainer, manager, train_test_split_kwargs)
    #     loader, submitter = self._create_test_loader_submitter(test_nrows)
    #     submitter.create_submission(manager, loader, **submitter_create_submission_kwargs)
    #
    # def _create_test_loader_submitter(self, test_nrows=None):
    #     test_container = DataContainer(path=self.test_path, nrows=test_nrows)
    #     test_dataset = SubmDataset(test_container)
    #     loader = self.data_assistant.get_without_split(test_dataset, has_answers=False)
    #     submitter = Submitter(**self.submitter_init_kwargs)
    #     return loader, submitter
    #
    # def cross_val(self, weights_updater_kwargs=None, model_kwargs=None, processor_kwargs=None,
    #               cross_val_kwargs=None) -> list:
    #     weights_updater_kwargs, model_kwargs, processor_kwargs, cross_val_kwargs = \
    #         self._set_to_dict_if_none(weights_updater_kwargs, model_kwargs, processor_kwargs, cross_val_kwargs)
    #     trainer = self._create_trainer(weights_updater_kwargs)
    #     manager = self._create_manager(model_kwargs, processor_kwargs)
    #     cross_validator = self._create_cross_validator()
    #
    #     cross_val_res = cross_validator.run(trainer, manager, self.train_dataset, self.data_assistant,
    #                                         fit_kwargs=self.trainer_fit_kwargs, **cross_val_kwargs)
    #     return cross_val_res
    #
    # def _create_cross_validator(self):
    #     return CrossValidator(**self.cross_val_init_kwargs)
    #
    #
    # def _fit(self, trainer, manager, train_test_split_kwargs):
    #     train_loader, val_loader = self.data_assistant.train_test_split(self.train_dataset, **train_test_split_kwargs)
    #     trainer.fit(train_loader, val_loader, manager, **self.trainer_fit_kwargs)
    #
    #
    def _set_to_dict_if_none(self, *args: Iterable[Union[None, dict]]) -> Collection:
        returned = []
        for arg in args:
            if arg is None:
                arg = {}
            returned.append(arg)
        return returned
    #
    def _create_manager(self, run_model_kwargs, run_proc_kwargs) -> ModelManager:
        model_kwargs = self.std_model_kwargs.copy()
        model_kwargs.update(run_model_kwargs)
        proc_kwargs = self.std_processor_kwargs.copy()
        proc_kwargs.update(run_proc_kwargs)

        model = SentPairBinaryClassifier(self.mname, **model_kwargs)
        proc = RucosProcessor(self.mname, **proc_kwargs)

        return ModelManager(model, proc, device=self.device)
    #
    def _create_trainer(self, run_weights_updater_kwargs) -> Trainer:
        weights_update_kwargs = self.std_weights_kwargs.copy()
        print("_create_trainer", run_weights_updater_kwargs, weights_update_kwargs)
        weights_update_kwargs.update(run_weights_updater_kwargs)

        weights_updater = QAWeightsUpdater(**weights_update_kwargs)
        return Trainer(self.validator, weights_updater, self.saver)
