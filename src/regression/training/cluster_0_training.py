from itertools import product
import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
from src.regression.train_model_for_cluster import register_trainer

logger = logging.getLogger(__name__)

@register_trainer(0)
def cluster_0_training(data_path: str) -> str:
    try:
        logger.info("Reading dataset from %s", data_path)
        dataset = pd.read_csv(data_path)
        X = dataset.drop(columns=['Bankrupt?']).values
        y = dataset['Bankrupt?'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=33, shuffle=True, stratify=y
        )

        group_train = [X_train.shape[0]]
        group_test = [X_test.shape[0]]

        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'n_estimators': [100, 200],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

        best_score = -np.inf
        best_model = None
        best_params = {}

        logger.info("Starting manual grid search over hyperparameters...")
        for lr, md, ne, ss, cs in product(
            param_grid['learning_rate'],
            param_grid['max_depth'],
            param_grid['n_estimators'],
            param_grid['subsample'],
            param_grid['colsample_bytree']
        ):
            model = XGBRanker(
                objective='rank:pairwise',
                learning_rate=lr,
                max_depth=md,
                n_estimators=ne,
                subsample=ss,
                colsample_bytree=cs,
                random_state=42,
                eval_metric='ndcg'
            )

            model.fit(
                X_train, y_train,
                group=group_train,
                eval_set=[(X_test, y_test)],
                eval_group=[group_test],
                verbose=False
            )

            y_pred = model.predict(X_test)
            score = average_precision_score(y_test, y_pred)

            logger.info(f"Evaluated config: LR={lr}, MaxDepth={md}, Estimators={ne} → AP Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = {
                    'learning_rate': lr,
                    'max_depth': md,
                    'n_estimators': ne,
                    'subsample': ss,
                    'colsample_bytree': cs
                }

        logger.info("Best score: %.4f with parameters: %s", best_score, best_params)

        # 🚀 MLflow Tracking
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("cluster-0-regression-model-training")

        with mlflow.start_run(run_name="cluster-0-regression-model-run") as run:
            run_id = run.info.run_id

            mlflow.set_tags({
                "model_type": "XGBRanker",
                "cluster": "0",
                "objective": "rank:pairwise"
            })

            logger.info("Logging best model parameters to MLflow...")
            mlflow.log_params(best_params)

            logger.info("Evaluating best model on test set for final metrics...")
            y_scores = best_model.predict(X_test)
            final_score = average_precision_score(y_test, y_scores)
            mlflow.log_metric("average_precision_score", final_score)

            model_name='cluster_0_regression_model'
            logger.info("Logging model artifact to MLflow...")
            mlflow.sklearn.log_model(best_model, artifact_path=model_name)
        
            client = mlflow.MlflowClient()
            model_uri = f"runs:/{run_id}/cluster_0_regression_model"
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


            logger.info("Model logged successfully at: %s", model_uri)

            return model_uri

    except Exception as e:
        logger.exception("Exception occurred during cluster 0 model training")
        raise
