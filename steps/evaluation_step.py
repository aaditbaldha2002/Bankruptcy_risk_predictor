import logging
from typing import List

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from zenml.steps import step
import mlflow

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def evaluation_step(data_path: str) -> None:
    try:
        logger.info("Loading test dataset...")
        df = pd.read_csv(data_path)

        if 'Cluster' not in df.columns or 'Bankrupt?' not in df.columns:
            raise ValueError("Missing 'Cluster' or 'Bankrupt?' column in test data.")

        X = df.drop(columns=['Cluster', 'Bankrupt?', 'Index'])
        y_class = df['Cluster']
        y_regression = df['Bankrupt?']

        logger.info("Running classification evaluation...")
        y_class_pred = mlflow..predict(X)
        acc = accuracy_score(y_class, y_class_pred)
        prec = precision_score(y_class, y_class_pred, average='macro')
        rec = recall_score(y_class, y_class_pred, average='macro')
        f1 = f1_score(y_class, y_class_pred, average='macro')

        with mlflow.start_run(run_name="evaluation_classification", nested=True):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_macro", f1)
            logger.info(f"Classification Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        logger.info("Running regression evaluation per cluster...")
        for cluster_id, model in enumerate(regressor_models):
            cluster_df = df[df['Cluster'] == cluster_id]
            if cluster_df.empty:
                logger.warning(f"No test data for cluster {cluster_id}, skipping...")
                continue

            X_cluster = cluster_df.drop(columns=['Cluster', 'Bankrupt?', 'Index'])
            y_true = cluster_df['Bankrupt?']
            y_pred = model.predict(X_cluster)

            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            with mlflow.start_run(run_name=f"evaluation_regression_cluster_{cluster_id}", nested=True):
                mlflow.log_param("cluster_id", cluster_id)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2_score", r2)

            logger.info(f"Cluster {cluster_id} - MSE: {mse:.4f}, R2: {r2:.4f}")

    except Exception as e:
        logger.exception(f"Evaluation step failed: {e}")
        raise
