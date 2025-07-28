import logging
import numpy as np
from zenml.steps import step
from src.inferencing.predict_on_cluster_label import predict_on_cluster_label

from src.inferencing.cluster_0_prediction import cluster_0_prediction
from src.inferencing.cluster_1_prediction import cluster_1_prediction
from src.inferencing.cluster_2_prediction import cluster_2_prediction
from src.inferencing.cluster_3_prediction import cluster_3_prediction
from src.inferencing.cluster_4_prediction import cluster_4_prediction


@step(enable_cache=True)
def predict_bankruptcy_step(transformed_data_file_path:str,cluster_label:int)->int:
    try:
        logging.info("Starting the predict_bankruptcy_step step ...")
        final_prediction=predict_on_cluster_label(transformed_data_file_path,cluster_label)
        logging.info("predict_bankruptcy_step completed")
        logging.info(f"Final prediction:{final_prediction}")
        return final_prediction
    except Exception as e:
        logging.error(f"Error occurred in predict_bankruptcy_step step:{e}")
        raise e
    pass