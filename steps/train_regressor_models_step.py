import logging
from typing import List

import pandas as pd
from sklearn.base import RegressorMixin
from zenml.steps import step

from src.regression.preprocessing.cluster_0_preprocessing import cluster_0_preprocessing
from src.regression.preprocessing.cluster_1_preprocessing import cluster_1_preprocessing
from src.regression.preprocessing.cluster_2_preprocessing import cluster_2_preprocessing
from src.regression.preprocessing.cluster_3_preprocessing import cluster_3_preprocessing
from src.regression.preprocessing.cluster_4_preprocessing import cluster_4_preprocessing

from src.regression.training.cluster_0_training import cluster_0_training
from src.regression.training.cluster_1_training import cluster_1_training
from src.regression.training.cluster_2_training import cluster_2_training
from src.regression.training.cluster_3_training import cluster_3_training
from src.regression.training.cluster_4_training import cluster_4_training

from src.regression.preprocess_cluster_data import preprocess_cluster_data
from src.regression.train_model_for_cluster import train_model_for_cluster

@step(enable_cache=False)
def train_regressor_model_step(data_paths:List[str])->List[str]:
    logging.info("Starting regressor models training step ...")
    for cluster_id,data_path in enumerate(data_paths):
        logging.info(f"🔍 Preprocessing cluster {cluster_id}")
        preprocessed_cluster_data_path=preprocess_cluster_data(cluster_id,data_path)
        model_uris=train_model_for_cluster(cluster_id,preprocessed_cluster_data_path)
    logging.info("Training regressor model step completed.")
    return model_uris