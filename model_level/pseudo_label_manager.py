from model_level.models.transformer_qanda import TransformerQA
from .processors.qa_pseudo_label_processor import QAPseudoLabelProcessor

from .types import *

class PseudoLabelModelManager:
    # TODO write type of processor
    def __init__(self, model: TransformerQA, processor: QAPseudoLabelProcessor,
                 device:Optional[torch.device]=None):

        self.model = model
        self.processor = processor
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

    def preproc_forward(self, features: UnprocFeatures) -> ModelPreds:
        out = self.processor.preprocess(features, device=self.device)
        preds = self.model(out)
        return preds

    def predict_postproc(self, features: UnprocFeatures) -> Tuple[PredAnswerCharIdxs, PredProbs]:
        model_preds = self.preproc_forward(features)
        (start_token_idxs, end_token_idxs), pred_probs = self.processor.best_candidates(model_preds)
        start_char_idxs, end_char_idxs = self.processor.token_start_end_to_char_idxs(start_token_idxs, end_token_idxs, features)
        start_end_char_idxs = list(zip(start_char_idxs, end_char_idxs))
        return start_end_char_idxs, pred_probs
