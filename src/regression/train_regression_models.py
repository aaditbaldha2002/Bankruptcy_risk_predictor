from itertools import product
import logging
from typing import List

import mlflow
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker

from preprocessing.pca_feature_reduction import hybrid_iterative_reduction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def precision_at_k(y_true, y_scores, k):
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[top_k_idx]) / k

def train_regression_models(data_paths: List[str]) -> List[RegressorMixin]:
    regressor_models_list = []

    for i, data_path in enumerate(data_paths):
        try:
            logger.info(f"📦 Processing data path {i}: {data_path}")

            # Load dataset
            df = pd.read_csv(data_path)
            indexes = df['Index']
            bankrupt_ = df['Bankrupt?']

            # Drop known noisy columns
            df = df.drop(columns=[
                'Cash/Total Assets', 'Total debt/Total net worth', 'Equity to Long-term Liability',
                'Cash/Current Liability', 'Long-term Liability to Current Assets', 'Quick Ratio',
                'Working capitcal Turnover Rate', 'Current Ratio', 'Quick Assets/Current Liability'
            ])

            logger.info("🔄 Scaling data...")
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled_array, columns=df.columns)

            logger.info("🧠 Performing PCA-based reduction...")
            final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
                scaled_df, thresh_low=0.9, thresh_high=0.95, verbose=True
            )

            X = final_df.values
            y = bankrupt_.values

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

            logger.info("🔍 Starting hyperparameter sweep...")

            for lr, md, ne, ss, cs in product(
                param_grid['learning_rate'],
                param_grid['max_depth'],
                param_grid['n_estimators'],
                param_grid['subsample'],
                param_grid['colsample_bytree']
            ):
                with mlflow.start_run(run_name=f"xgboost_cluster_{i}", nested=True):
                    mlflow.log_param("cluster_id", i)
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("max_depth", md)
                    mlflow.log_param("n_estimators", ne)
                    mlflow.log_param("subsample", ss)
                    mlflow.log_param("colsample_bytree", cs)

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

                    model.fit(X_train, y_train, group=group_train,
                              eval_set=[(X_test, y_test)],
                              eval_group=[group_test],
                              verbose=False)

                    y_pred = model.predict(X_test)
                    score = average_precision_score(y_test, y_pred)

                    mlflow.log_metric("average_precision", score)

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

                        logger.info(f"✅ New best AP: {score:.4f} with params: {best_params}")

            if best_model:
                logger.info("📊 Logging final metrics...")
                mlflow.set_tag("best_model_for_cluster", str(i))
                mlflow.log_params(best_params)
                mlflow.log_metric("best_average_precision", best_score)

                y_scores = best_model.predict(X_test)
                for k in [5, 10, 20]:
                    prec_k = precision_at_k(y_test, y_scores, k)
                    mlflow.log_metric(f"precision_at_{k}", prec_k)
                    logger.info(f"🎯 Precision@{k}: {prec_k:.4f}")

                regressor_models_list.append(best_model)

        except Exception as e:
            logger.exception(f"❌ Exception occurred for data_path {data_path}: {e}")

    return regressor_models_list
