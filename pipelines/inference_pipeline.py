import logging
from typing import List
import numpy as np

from zenml.pipelines import pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd

from steps.cluster_predict_data_step import cluster_predict_data_step
from steps.inference_preprocess_data_step import inference_preprocess_data_step
from steps.predict_bankruptcy_step import predict_bankruptcy_step

@pipeline(enable_cache=True)
def inference_pipeline(data:List[float]) -> int:
    logging.info("Starting the inference pipeline...")
    logging.info("Starting preprocessing of the input data...")
    
    try:
        transformed_data_file_path=inference_preprocess_data_step(data)
    except Exception as e:
        logging.error(f"error while doing step inference_preprocess_data_step step:{e}")
        raise e
    
    logging.info("Preprocessing of input data completed")
    logging.info("Performing cluster prediction on the transformed data ...")
    try:
        cluster_label=cluster_predict_data_step(transformed_data_file_path)
    except Exception as e:
        logging.error(f"error while doing step cluster_predict_data_step step:{e}")
        raise e
    logging.info("cluster prediction of input data completed")
    logging.info("Performing the prediction based on the cluster predicted")
    try:
        final_prediction=predict_bankruptcy_step(transformed_data_file_path,cluster_label)
    except Exception as e:
        logging.error(f"Error occurred while trying to predict bankruptcy:{e}")
        raise e
    return final_prediction