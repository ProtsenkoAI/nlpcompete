from data.contain import DataContainer
from data.datasets.standard_dataset import StandardDataset
from data.loaders_creation import DataLoaderSepPartsBuilder

from model_level.saving.local_saver import LocalSaver
from model_level.models.transformer_qanda import TransformerQA
from model_level.processors import QADataProcessor
from model_level.evaluating import Validator
from model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
from model_level.managing_model import ModelManager

from pipeline_components.train import Trainer


class QATrainEvalPipeline:
    """Intended to define unchanging task-related pipeline values once and then
    do all operations providing needed parameters (esp. model and training architecture).
    Also supports standard kwargs passed to objects that can be redifined in run().
    """
    def __init__(self, train_path, transformer_mname, models_save_dir,
                 device, batch_size, trainer_fit_kwargs={},
                 weights_updater_standard_kwargs={},
                 model_standard_kwargs={},
                 processor_standard_kwargs={},
                 nrows=None):
        """Creates unchanging objects and saved params"""
        train_container = DataContainer(path=train_path, nrows=nrows)
        self.train_dataset = StandardDataset(train_container)

        self.loader_builder = DataLoaderSepPartsBuilder(batch=batch_size)
        self.validator = Validator(self.loader_builder)
        self.saver = LocalSaver(models_save_dir)

        self.trainer_fit_kwargs = trainer_fit_kwargs
        self.mname = transformer_mname
        self.device = device
        self.std_model_kwargs = model_standard_kwargs
        self.std_processor_kwargs = processor_standard_kwargs
        self.std_weights_kwargs = weights_updater_standard_kwargs

    def run(self, weights_updater_kwargs, model_kwargs, processor_kwargs):
        manager = self._create_manager(model_kwargs, processor_kwargs)
        trainer = self._create_trainer(weights_updater_kwargs)
        trainer.fit(self.train_dataset, manager, **self.trainer_fit_kwargs)
        val_scores = trainer.get_eval_vals()
        return val_scores

    def _create_manager(self, run_model_kwargs, run_proc_kwargs):
        model_kwargs = self.std_model_kwargs.copy()
        model_kwargs.update(run_model_kwargs)
        proc_kwargs = self.std_processor_kwargs.copy()
        proc_kwargs.update(run_proc_kwargs)

        model = TransformerQA(self.mname, **model_kwargs)
        proc = QADataProcessor(self.mname, **proc_kwargs)

        return ModelManager(model, proc, device=self.device)

    def _create_trainer(self, run_weights_updater_kwargs):
        weights_update_kwargs = self.std_weights_kwargs.copy()
        weights_update_kwargs.update(run_weights_updater_kwargs)

        weights_updater = QAWeightsUpdater(**weights_update_kwargs)
        return Trainer(self.validator, self.loader_builder, weights_updater, self.saver)
