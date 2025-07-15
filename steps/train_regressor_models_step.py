import logging
from typing import List

import pandas as pd
from sklearn.base import RegressorMixin
from zenml.steps import step

@step(enable_cache=True)
def train_regressor_model_step(data_paths:List[str])->List[RegressorMixin]:
    
    pass