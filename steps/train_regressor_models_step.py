import logging
from typing import List

import pandas as pd
from sklearn.base import RegressorMixin
from zenml.steps import step

from regression.train_regression_models import train_regression_models

@step(enable_cache=True)
def train_regressor_model_step(data_paths:List[str])->List[str]:
    logging.info("Starting regressor models training step ...")
    model_uris=train_regression_models(data_paths)
    logging.info("Training regressor model step completed.")
    return model_uris