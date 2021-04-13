from transformers import BertForMaskedLM, BertTokenizer
from pipeline.data import BaseContainer


class MLMPretrainer:
    def pretrain(self, model: BertForMaskedLM, data_container: BaseContainer, tokenizer: BertTokenizer,
                 checkpoints_dir: str, test_size: float = 0.2):
        # TODO
        raise NotImplementedError
