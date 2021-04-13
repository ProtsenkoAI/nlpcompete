from typing import List, Any
import pandas as pd
import json

from pipeline.submitting import Submitter


class RucosSubmitter(Submitter):
    # TODO: add tqdm

    SamplePrediction = Any
    SubmObj = Any

    def pred_batch(self, batch, manager) -> List[SamplePrediction]:
        batch_preds = manager.predict_postproc(batch)
        return batch_preds

    def form_submission(self, preds: List[SamplePrediction]) -> SubmObj:
        df = pd.DataFrame(data=preds)
        subm = []
        for idx, sub_df in df.groupby('idx'):
            sorted_df = sub_df.sort_values(by='probs', ascending=False)

            answer = sorted_df.iloc[0]
            subm.append({
                'idx': int(answer['idx']),
                'end': int(answer['end']),
                'start': int(answer['start']),
                'text': answer['placeholder'],
            })
        return subm

    def write_submission(self, subm: SubmObj, sub_path: str):
        with open(sub_path, 'w') as f:
            f.writelines(json.dumps(obj, ensure_ascii=False) + '\n' for obj in subm)
