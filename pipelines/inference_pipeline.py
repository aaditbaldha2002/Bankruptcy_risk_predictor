import logging
from typing import List
import numpy as np

from zenml.pipelines import pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd

from steps.cluster_predict_data_step import cluster_predict_data_step
from steps.inference_preprocess_data_step import inference_preprocess_data_step

@pipeline(enable_cache=True)
def inference_pipeline(data:List[float]) -> int:
    logging.info("Starting the inference pipeline...")
    logging.info("Starting preprocessing of the input data...")
    transformed_data_file_path=inference_preprocess_data_step(data)
    logging.info("Preprocessing of input data completed")
    logging.info("Performing cluster prediction on the transformed data ...")
    cluster_label=cluster_predict_data_step(transformed_data_file_path)

    return -1