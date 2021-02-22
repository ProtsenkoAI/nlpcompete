from pipeline_components.train import Trainer
from model_level.models.transformer_qanda import TransformerQA
from model_level.evaluating import Validator
from model_level.managing_model import ModelManager
from model_level.updating_weights.qa_weights_updater import QAWeightsUpdater
from data.contain import DataContainer
from data.loaders_creation import DataLoaderSepPartsBuilder
from data.datasets import StandardDataset, SubmDataset
from model_level.processors import QADataProcessor
from model_level.saving.local_saver import LocalSaver
from .config import TestsConfig

import torch
config = TestsConfig()


def get_trainer(weights_updater_kwargs={}, **trainer_kwargs):
    loader_builder = get_loader_builder()
    validator = get_validator()
    weights_updater = get_weights_updater(**weights_updater_kwargs)

    trainer = Trainer(validator, loader_builder, weights_updater)
    return trainer


def get_weights_updater(**kwargs):
    return QAWeightsUpdater(**kwargs)


def get_model(**model_kwargs):
    model = TransformerQA(mname=config.model_name, **model_kwargs)
    return model


def get_validator():
    loader_builder = get_loader_builder()
    return Validator(loader_builder)


def get_container(path=None, nrows=10):
    if path is None:
        path = config.train_path
    return DataContainer(path, nrows=nrows)


def get_train_dataset(container=None, nrows=10):
    if container is None:
        container = get_container(nrows=nrows)
    return StandardDataset(container)
    

def get_val_dataset(nrows=10):
    return get_train_dataset(nrows=nrows)


def get_subm_dataset(container=None, nrows=10):
    if container is None:
        container = get_container(nrows=nrows)
    return SubmDataset(container)


def get_loader_builder():
    return DataLoaderSepPartsBuilder(config.batch_size)


def get_loader(dataset=None):
    if dataset is None:
        dataset = get_train_dataset()
    loader = get_loader_builder().build(dataset)
    return loader


def get_qa_processor(mname="DeepPavlov/rubert-base-cased"):
    return QADataProcessor(mname)


def get_model_manager(model=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = get_model()
    data_processor = get_qa_processor()
    
    return ModelManager(model, data_processor, device=device)


def get_local_saver(**kwargs):
    return LocalSaver(**kwargs)
