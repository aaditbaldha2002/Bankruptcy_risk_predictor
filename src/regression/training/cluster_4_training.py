import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

logger = logging.getLogger(__name__)

@register_trainer(4)
def cluster_4_training(data_path: str) -> str:
    try:
        logger.info("Reading dataset from: %s", data_path)
        dataset = pd.read_csv(data_path)
        X = dataset.drop(columns=['Bankrupt?'])
        y = dataset['Bankrupt?'].values

        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info("Initializing base XGBoost model...")
        base_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            learning_rate=0.01,
            max_depth=3,
            n_estimators=50,
            colsample_bytree=1.0,
            subsample=1.0,
            random_state=42,
            n_jobs=-1
        )

        logger.info("Starting Stratified K-Fold with SMOTE...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros_like(y, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info("Applying SMOTE on fold %d", fold)
            X_resampled, y_resampled = SMOTE(random_state=42, n_jobs=-1).fit_resample(X_train, y_train)

            model = clone(base_model)
            model.fit(X_resampled, y_resampled, verbose=False)
            y_probs[test_idx] = model.predict_proba(X_test)[:, 1]

        logger.info("Evaluating aggregated model predictions...")
        precision, recall, thresholds = precision_recall_curve(y, y_probs)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-10)
        best_idx = np.argmax(f2_scores[:-1])
        best_thresh = thresholds[best_idx]
        f2 = f2_scores[best_idx]
        avg_prec = average_precision_score(y, y_probs)

        y_pred = (y_probs >= best_thresh).astype(int)

        logger.info("Best Threshold: %.4f | F2 Score: %.4f | Avg Precision: %.4f", best_thresh, f2, avg_prec)

        logger.info("Logging to MLflow...")
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("cluster-4-xgboost-kfold")

        with mlflow.start_run(run_name="XGB-CV-SMOTE-Cluster4") as run:
            run_id = run.info.run_id

            # Log metrics
            mlflow.log_metric("best_threshold", best_thresh)
            mlflow.log_metric("f2_score", f2)
            mlflow.log_metric("average_precision", avg_prec)

            # Log base model (trained once for artifact purposes)
            logger.info("Logging representative model to MLflow")
            final_model = clone(base_model)
            final_model.fit(X_scaled, y, verbose=False)
            mlflow.sklearn.log_model(final_model, "xgb_cluster4_model")

            model_uri = f"runs:/{run_id}/xgb_cluster4_model"
            logger.info("Model URI: %s", model_uri)
            return model_uri

    except Exception as e:
        logger.exception("Error in cluster_4_training: %s", str(e))
        raise
