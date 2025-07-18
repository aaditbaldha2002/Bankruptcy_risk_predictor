import logging
from typing import List

import pandas as pd
from sklearn.base import RegressorMixin
from zenml.steps import step

from src.regression.preprocess_cluster_data import preprocess_cluster_data
from src.regression.train_model_for_cluster import train_model_for_cluster

@step(enable_cache=True)
def train_regressor_model_step(data_paths:List[str])->List[str]:
    logging.info("Starting regressor models training step ...")
    for cluster_id,data_path in enumerate(data_paths):
        logging.info(f"🔍 Preprocessing cluster {cluster_id}")
        preprocessed_cluster_data_path=preprocess_cluster_data(cluster_id,data_path)
        model_uris=train_model_for_cluster(cluster_id,preprocessed_cluster_data_path)
    logging.info("Training regressor model step completed.")
    return model_uris