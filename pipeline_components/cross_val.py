from typing import Union


class CrossValidator:
    def run(self, trainer, manager, dataset, data_assistant, fit_kwargs: Union[None, dict]=None,
            nfolds=5, max_fold=3) -> list:
        folds_results = []
        for train_loader, val_loader in data_assistant.split_folds(dataset, nfolds=nfolds, max_fold=max_fold):
            trainer.fit(train_loader, val_loader, manager, **fit_kwargs)
            eval_vals = trainer.get_eval_vals()
            best_eval_val = max(eval_vals)
            folds_results.append(best_eval_val)
        return folds_results