from typing import Optional
import json

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import pandas as pd
import os

from model_level.managing_model import ModelManager


class RucosSubmitter:
    def __init__(self, subm_dir: Optional[str] = None):
        if subm_dir is None:
            subm_dir = '.'
        self.subm_dir =subm_dir

    def create_submission(self, mm: ModelManager, loader: DataLoader, subm_file_name: str = 'submission',
                          probs_file_name=None):
        df = self._get_placeholders_probs_dataframe(mm, loader)
        print("len of df", len(df))
        if not probs_file_name is None:
            df.to_csv(os.path.join(self.subm_dir, probs_file_name))
        subm = []
        for idx, sub_df in df.groupby('idx'):

            try:
                sorted = sub_df.sort_values(by='probs', ascending=False)
            except ValueError as e:
                # print("VALUE ERROR in sorting", e)
                # print("sorted", sub_df)
                sorted = sub_df

            answer = sorted.iloc[0]
            subm.append({
                'idx': int(answer['idx']),
                'end': int(answer['end']),
                'start': int(answer['start']),
                'text': answer['placeholder'],
            })
        if not subm_file_name is None:
            with open(os.path.join(self.subm_dir, subm_file_name), 'w') as f:
                f.writelines(json.dumps(bruh, ensure_ascii=False) + '\n' for bruh in subm)

    def _get_placeholders_probs_dataframe(self, manager: ModelManager, loader: DataLoader) -> pd.DataFrame:
        res = {
            'idx': [],
            'probs': [],
            'start': [],
            'end': [],
            'placeholder': []
        }
        for text1, text2, idx, start, end, placeholder in tqdm(loader):
            idx, probs, start, end, placeholder = manager.predict_postproc((text1, text2, idx, start, end, placeholder))
            res['idx'].extend(idx)
            res['probs'].extend(probs[:, 1].tolist())
            res['start'].extend(start)
            res['end'].extend(end)
            res['placeholder'].extend(placeholder)
        return pd.DataFrame(data=res)
