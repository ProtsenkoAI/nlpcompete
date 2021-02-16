from modeling.train import Trainer
from modeling.transformer_qanda import TransformerQA
from modeling.evaluating import Validator
from modeling.managing_model import ModelManager
from data.contain import DataContainer
from data.loaders_creation import DataLoaderSepXYCreator
from data.datasets import TrainDataset, EvalDataset
from data.processors import QADataProcessor
from .config import TestsConfig
config = TestsConfig()


def get_trainer(model=None, **trainer_kwargs):
    if model is None:
        model = get_model()
    validator = get_validator()
    device = config.device
    trainer = Trainer(model, validator, device, **trainer_kwargs)
    return trainer

def get_model(**model_kwargs):
    model = TransformerQA(mname=config.model_name, **model_kwargs)
    return model


def get_validator():
    return Validator()


def get_container():
    return DataContainer(config.train_path, nrows=10)


def get_train_dataset():
    container = get_container()
    return TrainDataset(container, config.model_name)
    

def get_val_dataset():
    container = get_container()
    return EvalDataset(container, config.model_name, has_answers=True)


def get_test_dataset():
    container = get_container()
    return EvalDataset(container, config.model_name, has_answers=False)


def get_loader(dataset=None):
    if dataset is None:
        dataset = get_train_dataset()

    return DataLoaderSepXYCreator(dataset, config.batch_size).get()

def get_model_manager(model=None):
    if model is None:
        model = get_model()
    data_processor = QADataProcessor("DeepPavlov/rubert-base-cased")
    
    return ModelManager(model, data_processor)
