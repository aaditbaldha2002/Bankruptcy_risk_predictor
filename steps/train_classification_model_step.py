import logging
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier
from zenml.steps import step
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import mlflow
import mlflow.sklearn

from src.classification.train_classification_model import train_classification_model


logger = logging.getLogger(__name__)

@step(enable_cache=True)
def train_classification_model_step(data_path: str) -> str:
    logger.info("Starting train_classification_model_step...")
    classifier_model_uri=train_classification_model(data_path)
    logger.info("train_classification_model_step completed.")
    return classifier_model_uri