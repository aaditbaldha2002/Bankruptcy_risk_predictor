import logging
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRanker
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.metrics import recall_score, precision_score, average_precision_score

from src.regression.train_model_for_cluster import register_trainer

logger = logging.getLogger(__name__)

@register_trainer(2)
def cluster_2_training(data_path: str) -> str:
    try:
        logger.info("Reading dataset from %s", data_path)
        dataset = pd.read_csv(data_path)
        X = dataset.drop(columns=['Bankrupt?'], axis=1)
        y = dataset['Bankrupt?']

        logger.info("Performing stratified train-test split...")
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logger.info("Standardizing data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_full)

        selected_features = set()

        logger.info("Extracting top features using Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train_full, y_train)
        selected_features.update(pd.Series(rf.feature_importances_, index=X.columns).nlargest(15).index.tolist())

        logger.info("Extracting top features using XGBoost...")
        xgb = XGBClassifier(
            use_label_encoder=False, eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() * 1.5,
            random_state=42
        )
        xgb.fit(X_train_full, y_train)
        selected_features.update(pd.Series(xgb.feature_importances_, index=X.columns).nlargest(15).index.tolist())

        logger.info("Extracting top features using MLP + Permutation Importance...")
        mlp = MLPClassifier(random_state=42, max_iter=500)
        mlp.fit(X_train_scaled, y_train)
        perm_result = permutation_importance(mlp, X_train_scaled, y_train, n_repeats=5, random_state=42)
        selected_features.update(pd.Series(perm_result.importances_mean, index=X.columns).nlargest(15).index.tolist())

        top_features = list(selected_features)
        X_train = X_train_full[top_features]
        X_test = X_test_full[top_features]

        logger.info("Setting up stacking classifier pipeline with SMOTEENN...")
        rf_final = RandomForestClassifier(random_state=42)
        mlp_final = MLPClassifier(random_state=42, max_iter=500)
        xgb_final = XGBClassifier(
            use_label_encoder=False, eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() * 1.5,
            random_state=42
        )
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)

        stack = StackingClassifier(
            estimators=[('rf', rf_final), ('mlp', mlp_final), ('xgb', xgb_final)],
            final_estimator=meta_learner,
            passthrough=True,
            n_jobs=-1
        )

        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('resampler', SMOTEENN(random_state=42)),
            ('stack', stack)
        ])

        logger.info("Initializing randomized hyperparameter search...")
        param_grid = {
            'stack__rf__n_estimators': [100, 200],
            'stack__rf__max_depth': [5, 10],
            'stack__mlp__hidden_layer_sizes': [(100,), (150,)],
            'stack__mlp__learning_rate_init': [0.001, 0.01],
            'stack__xgb__n_estimators': [100, 200],
            'stack__xgb__learning_rate': [0.01, 0.1],
            'stack__xgb__max_depth': [3, 5],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=20,
            scoring='recall',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        logger.info("Fitting best model...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        logger.info("Predicting bankruptcy scores...")
        y_scores = best_model.predict_proba(X_test)[:, 1]
        X_test_df = pd.DataFrame(X_test, columns=top_features)
        X_test_df['score'] = y_scores
        X_test_df['true_label'] = y_test.values
        X_test_df_sorted = X_test_df.sort_values('score', ascending=False).reset_index(drop=True)

        top_n = round(0.20 * len(X_test_df_sorted))
        X_test_df_sorted['predicted'] = 0
        X_test_df_sorted.loc[:top_n - 1, 'predicted'] = 1

        precision_topk = precision_score(X_test_df_sorted['true_label'], X_test_df_sorted['predicted'])
        recall_topk = recall_score(X_test_df_sorted['true_label'], X_test_df_sorted['predicted'])

        logger.info("Training XGBRanker model for average_precision_score metric...")

        # For XGBRanker, we need group info; here, as a proxy, assume one big group (or adapt as needed)
        group_train = [len(y_train)]  # single group
        group_test = [len(y_test)]    # single group

        xgbranker = XGBRanker(
            objective='rank:pairwise',
            random_state=42,
            eval_metric='map',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() * 1.5,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        xgbranker.fit(X_train, y_train, group=group_train)

        # Predict scores on test set
        y_scores_ranker = xgbranker.predict(X_test)
        aps_ranker = average_precision_score(y_test, y_scores_ranker)
        logger.info("XGBRanker average_precision_score: %.4f", aps_ranker)

        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        mlflow.set_experiment("cluster-2-stacked-bankruptcy-prediction")

        with mlflow.start_run(run_name="Stacked-Ensemble-Cluster2") as run:
            run_id = run.info.run_id
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("precision_topk", precision_topk)
            mlflow.log_metric("recall_topk", recall_topk)
            mlflow.log_metric("average_precision_score", aps_ranker)

            mlflow.sklearn.log_model(best_model, "stacked_model_cluster2")
            model_uri = f"runs:/{run_id}/stacked_model_cluster2"
            logger.info("Model saved at: %s", model_uri)

            return model_uri

    except Exception as e:
        logger.exception("Error during cluster 2 model training")
        raise
