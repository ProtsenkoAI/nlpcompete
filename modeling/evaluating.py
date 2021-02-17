from sklearn.metrics import f1_score
from tqdm.notebook import tqdm


class Validator:
    def __init__(self):
        # TODO: metrics are hardcoded now, maybe later we'll need to get them in arguments  
        # TODO: add multiple metrics
        self.metric = f1_score

    def eval(self, manager, test):
        print("starting to evaludate")
        manager.get_model().eval()
        preds, labels = self._pred_all_batches(manager, test)
        print("preds and labels", preds, labels)
        metric_val = self.metric(preds, labels)

        print(f"Example of eval probs: {preds[:20]}")
        print("Eval value:", metric_val)
        return metric_val

    def _pred_all_batches(self, manager, test):
        all_preds = []
        all_labels = []
        for batch in tqdm(test, desc="eval"):
            features, labels_unproc = batch
            preds, labels = manager.predict_postproc_labeled(features, labels_unproc)

            all_preds += list(preds)
            all_labels += list(labels)
        print("all_preds", all_preds)
        print("all_labels", all_labels  )
        return all_preds, all_labels
        

    # def _em_score(self, preds, correct_answers):
    #     raise NotImplementedError
