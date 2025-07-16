import logging
import pandas as pd
from zenml.pipelines import pipeline

from steps.cluster_prediction_step import cluster_prediction_step
from steps.evaluation_step import evaluation_step
from steps.preprocess_step import preprocess_step
from steps.train_classification_model_step import train_classification_model_step
from steps.train_regressor_models_step import train_regressor_model_step

@pipeline(enable_cache=True)
def deployment_pipeline(train_data_path: str, test_data_path: str) -> None:
    """
    Pipeline used for deployment of all the models

    Args:
        train_data_path(str): the file path for the training dataset
        test_data_path(str): the file path for the testing dataset
    """
    
    try:
        clean_data_path = preprocess_step(train_data_path)
    except Exception as e:
        logging.error(f"[Step: Preprocess] Failed to preprocess training data: {e}", exc_info=True)
        raise

    # try:
    #     cluster_labeled_file_path,cluster_file_paths = cluster_prediction_step(clean_data_path)
    # except Exception as e:
    #     logging.error(f"[Step: Cluster Prediction] Failed to cluster data: {e}", exc_info=True)
    #     raise

    # try:
    #     classifier_model_uri=train_classification_model_step(cluster_labeled_file_path)
    # except Exception as e:
    #     logging.error(f"[Step: Train Classification] Failed to train classification model: {e}", exc_info=True)
    #     raise

    # try:
    #     regressor_model_uris=train_regressor_model_step(cluster_file_paths)
    # except Exception as e:
    #     logging.error(f"[Step: Train Regression Models] Failed to train regression models: {e}", exc_info=True)
    #     raise

    # try:
    #     metrics = evaluation_step(test_data_path,classifier_model_uri,regressor_model_uris)
    # except Exception as e:
    #     logging.error(f"[Step: Evaluation] Failed to evaluate models: {e}", exc_info=True)
    #     raise

    # based on the metrics deploy the models
