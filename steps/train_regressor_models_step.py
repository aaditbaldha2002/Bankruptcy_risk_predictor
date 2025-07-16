import logging
from typing import List

import pandas as pd
from sklearn.base import RegressorMixin
from zenml.steps import step

@step(enable_cache=True)
def train_regressor_model_step(data_paths:List[str])->List[str]:
    return['test1','test2','test3','test4','test5']