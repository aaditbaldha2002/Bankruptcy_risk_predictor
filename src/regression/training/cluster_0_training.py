from itertools import product
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
from src.regression.train_model_for_cluster import register_trainer


@register_trainer(0)
def cluster_0_training(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    X = dataset.drop(columns=['Bankrupt?']).values
    y = dataset['Bankrupt?'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=33, shuffle=True, stratify=y
    )

    group_train = [X_train.shape[0]]  # One group
    group_test = [X_test.shape[0]]    # One group

    # 🧪 Step 2: Hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6],
        'n_estimators': [100, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    # 📊 Step 3: Manual Grid Search
    best_score = -np.inf
    best_model = None
    best_params = {}

    for lr, md, ne, ss, cs in product(
            param_grid['learning_rate'],
            param_grid['max_depth'],
            param_grid['n_estimators'],
            param_grid['subsample'],
            param_grid['colsample_bytree']):

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

    # ✅ Step 4: Final evaluation with best model
    y_scores = best_model.predict(X_test)
    score = average_precision_score(y_test, y_pred)

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
