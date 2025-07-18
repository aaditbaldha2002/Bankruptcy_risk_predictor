import logging
from typing import List

import mlflow
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score,accuracy_score, r2_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate_models(data_path:str,classifier_model_uri:str,regressor_model_uris:List[str]):
    try:
        logger.info("📦 Loading test dataset...")
        df = pd.read_csv(data_path)

        if 'Cluster' not in df.columns or 'Bankrupt?' not in df.columns:
            raise ValueError("Missing 'Cluster' or 'Bankrupt?' column in test data.")

        drop_cols = ['Cluster', 'Bankrupt?']
        if 'Index' in df.columns:
            drop_cols.append('Index')

        X = df.drop(columns=drop_cols)
        y_class = df['Cluster']
        y_regression = df['Bankrupt?']

        logger.info("🧠 Running classification model evaluation...")
        classifier_model = mlflow.sklearn.load_model(classifier_model_uri)
        y_class_pred = classifier_model.predict(X)

        acc = accuracy_score(y_class, y_class_pred)
        prec = precision_score(y_class, y_class_pred, average='macro')
        rec = recall_score(y_class, y_class_pred, average='macro')
        f1 = f1_score(y_class, y_class_pred, average='macro')

        with mlflow.start_run(run_name="evaluation_classification", nested=True):
            mlflow.set_tag("model_type", "classifier")
            mlflow.set_tag("model_uri", classifier_model_uri)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_macro", f1)

        logger.info(
            f"✅ Classification Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        logger.info("🔁 Running regression evaluation for each cluster...")
        for cluster_id, model_uri in enumerate(regressor_model_uris):
            cluster_df = df[df['Cluster'] == cluster_id]
            if cluster_df.empty:
                logger.warning(f"⚠️ No test data for cluster {cluster_id}, skipping...")
                continue

            cluster_drop_cols = ['Cluster', 'Bankrupt?']
            if 'Index' in cluster_df.columns:
                cluster_drop_cols.append('Index')

            X_cluster = cluster_df.drop(columns=cluster_drop_cols)
            y_true = cluster_df['Bankrupt?']

            regressor_model = mlflow.sklearn.load_model(model_uri)
            y_pred = regressor_model.predict(X_cluster)

            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            with mlflow.start_run(run_name=f"evaluation_regression_cluster_{cluster_id}", nested=True):
                mlflow.set_tag("model_type", "regressor")
                mlflow.set_tag("model_uri", model_uri)
                mlflow.log_param("cluster_id", cluster_id)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2_score", r2)

            logger.info(f"📈 Cluster {cluster_id} - MSE: {mse:.4f}, R2: {r2:.4f}")

    except Exception as e:
        logger.exception(f"❌ Evaluation step failed: {e}")
        raise
