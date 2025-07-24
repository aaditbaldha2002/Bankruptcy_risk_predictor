import logging
from typing import List
import numpy as np

from zenml.pipelines import pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd

@pipeline(enable_cache=True)
def inference_pipeline(data:List[float]) -> List[float]:
    df_train = pd.read_csv('../data/raw/train_data.csv')
    df_train_columns=df_train.columns.to_list()
    df = pd.DataFrame([data], columns=[f'{col_name}' for col_name in df_train_columns])
    

    pass