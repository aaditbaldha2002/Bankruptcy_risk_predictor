import logging
from typing import List

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from zenml.steps import step
import mlflow

from src.evaluation.evaluate_metrics import evaluate_metrics
from src.evaluation.evaluate_models import evaluate_models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@step(enable_cache=True)
def evaluation_step(
    data_path: str,
    classifier_model_uri: str,
    regressor_model_uris: List[str]
) -> bool:
    logger.info("Starting the evaluation step...")
    evaluate_models(data_path,classifier_model_uri,regressor_model_uris)
    logger.info("Evaluation step complemented.")
    logger.info("Evaluating model metrics...")
    deployment_decision=evaluate_metrics()
    logger.info("Evaluating model completed")
    return deployment_decision