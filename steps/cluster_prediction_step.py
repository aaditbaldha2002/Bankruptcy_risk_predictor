import logging
from typing import List

import pandas as pd
from zenml.steps import step

@step(enable_cache=True)
def cluster_prediction_step(data_path:str)->List[str]:
    pass