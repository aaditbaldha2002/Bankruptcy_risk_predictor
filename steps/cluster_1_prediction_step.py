import logging

import pandas as pd
from zenml.steps import step

@step(enable_cache=True)
def cluster_1_prediction_step():
    pass