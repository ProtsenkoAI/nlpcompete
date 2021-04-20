from sources.pipeline import Trainer
from sources.pipeline import QAPipeline
from sources.pipeline.model_selection.cross_val import CrossValidator
# from model_level.models.transformer_qanda import TransformerQA
from legacy.model_level.models import SentPairBinaryClassifier
from legacy.model_level import Validator
from sources.pipeline import ModelManager
from sources.pipeline import BlendingModelManager
from sources.pipeline import UnbatchingProcessor
from legacy.model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
from legacy.data import DataContainer
from legacy.data import DataLoaderSepPartsBuilder
from legacy.data import StandardDataset, SubmDataset
from legacy.data import DataAssistant
from legacy.model_level import RucosProcessor
from sources.pipeline.saving import LocalSaver
from .config import TestsConfig

import torch
from torch.utils.data import DataLoader
from typing import Optional
config = TestsConfig()


def get_trainer(weights_updater_kwargs:Optional[dict]=None) -> Trainer:
    if weights_updater_kwargs is None:
        weights_updater_kwargs = {}
    validator = get_validator()
    weights_updater = get_weights_updater(**weights_updater_kwargs)
    saver = get_local_saver()

    trainer = Trainer(validator, weights_updater, saver)
    return trainer


def get_weights_updater(**kwargs) -> QAWeightsUpdater:
    return QAWeightsUpdater(**kwargs)


def get_model(**model_kwargs) -> SentPairBinaryClassifier:
    model = SentPairBinaryClassifier(mname=config.model_name, **model_kwargs)
    return model


def get_validator() -> Validator:
    return Validator()


def get_container(path=None, nrows=10) -> DataContainer:
    if path is None:
        path = config.train_path
    return DataContainer(path, nrows=nrows)


def get_train_dataset(container=None, nrows=10) -> StandardDataset:
    if container is None:
        container = get_container(nrows=nrows)
    return StandardDataset(container)
    

def get_val_dataset(nrows=10) -> StandardDataset:
    return get_train_dataset(nrows=nrows)


def get_subm_dataset(container=None, nrows=10) -> SubmDataset:
    if container is None:
        container = get_container(nrows=nrows)
    return SubmDataset(container)


def get_loader_builder(batch_size=config.batch_size) -> DataLoaderSepPartsBuilder:
    return DataLoaderSepPartsBuilder(batch_size)


def get_loader(dataset=None, has_answers=True) -> DataLoader:
    if dataset is None:
        dataset = get_train_dataset()
    loader = get_loader_builder().build(dataset, has_answers=has_answers)
    return loader


# def get_qa_processor(mname=config.model_name) -> QADataProcessor:
#     return QADataProcessor(mname)

def get_rucos_processor() -> RucosProcessor:
    return RucosProcessor(mname=config.model_name)


def get_model_manager(model=None, device=None) -> ModelManager:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = get_model()
    data_processor = get_rucos_processor()
    
    return ModelManager(model, data_processor, device=device)

# def get_pseudo_label_manager(model=None, device=None) -> ModelManager:
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if model is None:
#         model = get_model()
#     processor = get_qa_pseudo_label_processor()
#     return ModelManager(model, processor, device=device)

# def get_qa_pseudo_label_processor() -> QAPseudoLabelProcessor:
#     return QAPseudoLabelProcessor(mname=config.model_name)


def get_local_saver(**kwargs) -> LocalSaver:
    return LocalSaver(**kwargs)


def get_qa_pipeline() -> QAPipeline:
    pipeline = QAPipeline(config.train_path, config.model_name, config.save_dir,
                          config.device, config.batch_size,
                          trainer_fit_kwargs={"max_step": 2, "steps_betw_evals": 1},
                          weights_updater_standard_kwargs={},
                          model_standard_kwargs={},
                          nrows=2)
    return pipeline

def get_data_assistant() -> DataAssistant:
    loader_creator = get_loader_builder()
    return DataAssistant(loader_creator)

def get_cross_validator() -> CrossValidator:
    return CrossValidator()

def get_unbatching_processor() -> UnbatchingProcessor:
    return UnbatchingProcessor()

def get_blending_model_manager(**kwargs) -> BlendingModelManager:
    return BlendingModelManager(**kwargs)
