from sklearn.metrics import f1_score
from tqdm.notebook import tqdm


class Validator:
    def __init__(self):
        self.main_metric = "f1"
        self.metrics = {"f1": f1_score, "em": self._em_score}

    def eval(self, model, test):
        model.eval()
        preds, labels = self._pred_all_batches(model, test)

        eval_vals = {}
        for name, metric_func in self.metrics:
            metric_val = metric_func(preds, labels)
            eval_vals[name] = metric_val

        print(f"Example of eval probs: {preds[:20]}")
        print("Eval value:", metric_val)
        return eval_vals

    def _pred_all_batches(self, model, test):
        all_preds = []
        all_labels = []
        for batch in tqdm(test, desc="eval"):
            features, labels = batch
            pred_classes = model.predict_classes(features, labels)

            all_preds += list(pred_classes)
            all_labels += list(labels.cpu())
        
        return all_preds, all_labels
        

    def _em_score(self, preds, correct_answers):
        raise NotImplementedError
    