import pandas as pd
import numpy as np
import logging
import mlflow
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from src.regression.train_model_for_cluster import register_trainer

logger = logging.getLogger(__name__)

@register_trainer(1)
def cluster_1_training(data_path: str) -> str:
    try:
        logger.info("Reading dataset from %s", data_path)
        dataset = pd.read_csv(data_path)

        X = dataset.drop(["Bankrupt?"], axis=1)
        y = dataset["Bankrupt?"]

        logger.info("Splitting dataset into stratified train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logger.info("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info("Applying ADASYN for oversampling...")
        adasyn = ADASYN(random_state=42)
        X_train_res, y_train_res = adasyn.fit_resample(X_train_scaled, y_train)

        logger.info("Setting up hyperparameter grid...")
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [6, 9, 12, 15],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [10, 20, 26, 30],
            'min_child_weight': [1, 5, 10],
            'gamma': [0, 0.1, 0.3]
        }

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logger.info("Running GridSearchCV on oversampled data...")
        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='average_precision',
            cv=cv,
            verbose=2,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )

        grid.fit(X_train_res, y_train_res)

        best_model = grid.best_estimator_
        logger.info("Best model parameters: %s", grid.best_params_)

        logger.info("Starting MLflow tracking...")
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("cluster-1-regression-model-training")

        with mlflow.start_run(run_name="cluster-1-regression-model-run") as run:
            run_id = run.info.run_id

            # Log parameters
            mlflow.log_params(grid.best_params_)

            # Predict and compute scores
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            avg_precision = average_precision_score(y_test, y_prob)

            # Precision-Recall Curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            optimal_threshold = 0.65  # Could be auto-tuned here

            y_pred_thresh = (y_prob >= optimal_threshold).astype(int)

            # Log metrics
            mlflow.log_metric("average_precision_score", avg_precision)
            mlflow.log_metric("optimal_threshold", optimal_threshold)

            logger.info("Logging model to MLflow...")
            model_name='cluster_1_regression_model'
            mlflow.sklearn.log_model(best_model, model_name)

            model_uri = f"runs:/{run_id}/cluster_1_regression_model"
            client = mlflow.MlflowClient()
            try:
                client.get_registered_model(model_name)
            except Exception as e:
                client.create_registered_model(model_name)
                
            # Register a new model version pointing to the logged artifact
            client.create_model_version(
                name=model_name,
                source=model_uri,  # your logged model artifact URI
                run_id=run_id
            )


            logger.info("Model saved at: %s", model_uri)

            return model_uri

    except Exception as e:
        logger.exception("Error during cluster 1 model training")
        raise
