import logging
import os

import joblib
from zenml.steps import step
import numpy as np
from src.inferencing.cluster_predict_data import cluster_predict_data

@step(enable_cache=True)
def cluster_predict_data_step(data_path:str)->np.int32:
    try:
        logging.info("starting the cluster prediction step for input data ...")
        cluster_label=cluster_predict_data(data_path)
        logging.info("predicted the cluster for the input data successfully")
        return np.int32(0)
    except Exception as e:
        logging.error(f"Error occurred in cluster_predict_data_step step:{e}")
        raise e