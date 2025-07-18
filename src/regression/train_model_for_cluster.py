from itertools import product
import logging
from typing import Callable, List

import mlflow
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAINING_REGISTRY: dict[int, Callable[[str], str]] = {}
def register_trainer(cluster_id: int):
    def decorator(func: Callable[[str], str]):
        TRAINING_REGISTRY[cluster_id] = func
        return func
    return decorator

def train_model_for_cluster(cluster_id: int, data_path: str) -> str:
    if cluster_id not in TRAINING_REGISTRY:
        raise ValueError(f"No trainer registered for cluster {cluster_id}")
    logger.info(f"🧠 Training model for cluster {cluster_id} using {data_path}")
    return TRAINING_REGISTRY[cluster_id](data_path)