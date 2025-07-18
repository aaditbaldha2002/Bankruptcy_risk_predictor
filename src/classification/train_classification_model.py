import logging
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def train_classification_model(data_path: str) -> str:
    try:
        logger.info("Reading dataset from %s", data_path)
        df = pd.read_csv(data_path)

        # Initial split
        X = df.drop(columns=['Cluster'])
        y = df['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        logger.info("Initializing base XGBoost model...")
        base_model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(y.unique()),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )

        base_param_grid = {
            'n_estimators': [300, 400],
            'max_depth': [1, 2, 3, 5, 7],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.55, 0.6, 0.65],
            'colsample_bytree': [1.0]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logger.info("Performing base grid search...")
        base_search = GridSearchCV(base_model, base_param_grid, scoring='recall_macro', cv=cv, n_jobs=-1, verbose=1)
        base_search.fit(X_train, y_train)

        logger.info("Selecting top features based on importance...")
        feature_importances = pd.Series(base_search.best_estimator_.feature_importances_, index=X.columns)
        top_n = 28
        top_features = feature_importances.sort_values(ascending=False).head(top_n).index.tolist()

        top_df = df[top_features + ['Cluster']]
        X = top_df.drop(columns=['Cluster'])
        y = top_df['Cluster']

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        reduced_model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(y.unique()),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )

        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        logger.info("Setting MLflow experiment...")
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("classification-model-training")

        with mlflow.start_run(run_name="XGB-Classification-Tuned") as run:
            run_id = run.info.run_id
            mlflow.set_tags({
                "stage": "classification",
                "model_type": "XGBoost",
                "input_features": len(top_features),
                "developer": "aadit.baldha"
            })

            logger.info("Training reduced XGBoost model...")
            reduced_search = GridSearchCV(reduced_model, reduced_param_grid, scoring='recall_macro', cv=cv, n_jobs=-1, verbose=1)
            reduced_search.fit(X_train_r, y_train_r)

            logger.info("Cross-validating reduced model...")
            scores = cross_validate(
                reduced_search.best_estimator_, X, y,
                cv=cv,
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                n_jobs=-1
            )

            logger.info("Logging best parameters and metrics to MLflow...")
            mlflow.log_params(reduced_search.best_params_)
            mlflow.log_metrics({
                "accuracy": scores['test_accuracy'].mean(),
                "precision_macro": scores['test_precision_macro'].mean(),
                "recall_macro": scores['test_recall_macro'].mean(),
                "f1_macro": scores['test_f1_macro'].mean()
            })

            logger.info("Logging final model to MLflow...")
            mlflow.sklearn.log_model(reduced_search.best_estimator_, artifact_path="xgb_classifier_model")

            model_uri = f"runs:/{run_id}/xgb_classifier_model"
            logger.info("Model logged at URI: %s", model_uri)
            return model_uri

    except Exception as e:
        logger.exception("Exception during classification model training")
        raise
