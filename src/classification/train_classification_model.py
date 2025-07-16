import logging

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def train_classification_model(data_path:str)->str:
    try:
        logger.info("Reading dataset...")
        df = pd.read_csv(data_path)
        X = df.drop(columns=['Cluster'])
        y = df['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        logger.info("Initializing base XGBoost model...")
        classification_model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(y.unique()),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )

        param_grid = {
            'n_estimators': [300, 400],
            'max_depth': [1, 2, 3, 5, 7],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.55, 0.6, 0.65],
            'colsample_bytree': [1.0]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logger.info("Running grid search for base model...")
        grid_search = GridSearchCV(classification_model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info("Feature importance analysis...")
        feature_importances = pd.Series(
            grid_search.best_estimator_.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        top_n = 28
        top_features = feature_importances.head(top_n).index.tolist()
        top_n_df = df[top_features + ['Cluster']]
        X = top_n_df.drop(columns=['Cluster'])
        y = top_n_df['Cluster']

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        xgb_reduced = XGBClassifier(
            objective='multi:softmax',
            num_class=len(y.unique()),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )

        param_grid_r = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        grid_search_r = GridSearchCV(xgb_reduced, param_grid_r, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
        
        logger.info("Tracking with MLflow...")
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("classification-model-training")

        with mlflow.start_run(run_name="XGB-Classification-Tuned") as run:

            logger.info("Training reduced model...")
            grid_search_r.fit(X_train_r, y_train_r)
            logger.info("Evaluating performance...")

            scores = cross_validate(
                grid_search_r.best_estimator_, X, y,
                cv=cv, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], n_jobs=-1
            )

            # Log parameters
            mlflow.log_params(grid_search_r.best_params_)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": scores['test_accuracy'].mean(),
                "precision_macro": scores['test_precision_macro'].mean(),
                "recall_macro": scores['test_recall_macro'].mean(),
                "f1_macro": scores['test_f1_macro'].mean()
            })

            # Log model
            mlflow.sklearn.log_model(grid_search_r.best_estimator_, "xgb_classifier_model")
            run_id = run.info.run_id
            classifier_model_uri=f'runs:/{run_id}/classifier_model'

        logger.info("Training and tracking complete.")
        return classifier_model_uri

    except Exception as e:
        logger.error(f"Error in train_classification_model_step: {e}")
        raise