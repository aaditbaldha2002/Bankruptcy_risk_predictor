import logging
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)

@register_trainer(3)
def cluster_3_training(data_path: str) -> str:
    try:
        logger.info("Reading dataset from: %s", data_path)
        dataset = pd.read_csv(data_path)
        X = dataset.drop(columns=['Bankrupt?']).values
        y = dataset['Bankrupt?'].values

        logger.info("Splitting dataset with stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        logger.info("Setting up SMOTETomek → Scaler → XGBoost pipeline...")
        pipeline = ImbPipeline([
            ('resample', SMOTETomek(random_state=42)),
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=1,
                n_jobs=-1,
                random_state=42
            ))
        ])

        param_distributions = {
            'clf__n_estimators': randint(300, 600),
            'clf__max_depth': randint(3, 10),
            'clf__learning_rate': uniform(0.01, 0.3),
            'clf__min_child_weight': randint(1, 6),
            'clf__subsample': uniform(0.6, 0.4),
            'clf__colsample_bytree': uniform(0.6, 0.4)
        }

        logger.info("Initiating randomized hyperparameter search...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=30,
            scoring='average_precision',
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        logger.info("Training model...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        logger.info("Evaluating best model...")
        y_proba = best_model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-6)
        best_threshold = thresholds[np.argmax(f2_scores)]

        y_pred = (y_proba >= best_threshold).astype(int)
        avg_precision = average_precision_score(y_test, y_proba)
        best_f2 = np.max(f2_scores)

        logger.info("Best average precision: %.4f", avg_precision)
        logger.info("Best F2 score: %.4f", best_f2)
        logger.info("Best threshold: %.4f", best_threshold)

        logger.info("Logging to MLflow...")
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("cluster-3-xgboost-bankruptcy")

        with mlflow.start_run(run_name="XGB-SMOTETomek-Cluster3") as run:
            run_id = run.info.run_id
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({
                "average_precision_score": avg_precision,
                "f2_score": best_f2,
                "best_threshold": best_threshold
            })
            mlflow.sklearn.log_model(best_model, "xgb_model_cluster3")
            model_uri = f"runs:/{run_id}/xgb_model_cluster3"
            logger.info("Model saved at URI: %s", model_uri)
            return model_uri

    except Exception as e:
        logger.exception("Error during training for Cluster 3")
        raise
