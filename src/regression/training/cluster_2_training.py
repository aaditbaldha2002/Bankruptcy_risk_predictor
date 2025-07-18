import logging
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN


@register_trainer(2)
def cluster_2_training(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    X=dataset.drop(columns=['Index','Bankrupt?'],axis=1)
    y=dataset['Bankrupt?']

    # Stratified split to preserve class ratio
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------- Scale full data ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    # ---------- Feature selection from base models ----------
    selected_features = set()

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_full, y_train)
    rf_top_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15).index.tolist()
    selected_features.update(rf_top_features)

    # 2. XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() * 1.5,
                        random_state=42)
    xgb.fit(X_train_full, y_train)
    xgb_top_features = pd.Series(xgb.feature_importances_, index=X.columns).nlargest(15).index.tolist()
    selected_features.update(xgb_top_features)

    # 3. MLP + Permutation Importance
    mlp = MLPClassifier(random_state=42, max_iter=500)
    mlp.fit(X_train_scaled, y_train)
    perm_result = permutation_importance(mlp, X_train_scaled, y_train, n_repeats=5, random_state=42)
    mlp_top_features = pd.Series(perm_result.importances_mean, index=X.columns).nlargest(15).index.tolist()
    selected_features.update(mlp_top_features)

    # ---------- Reduce dataset to selected features ----------
    top_features = list(selected_features)
    X_train = X_train_full[top_features]
    X_test = X_test_full[top_features]

    # ---------- Base Models & Meta Learner ----------
    rf_final = RandomForestClassifier(random_state=42)
    mlp_final = MLPClassifier(random_state=42, max_iter=500)
    xgb_final = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() * 1.5,
                            random_state=42)
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    # ---------- Stacking Classifier ----------
    stack = StackingClassifier(
        estimators=[
            ('rf', rf_final),
            ('mlp', mlp_final),
            ('xgb', xgb_final)
        ],
        final_estimator=meta_learner,
        passthrough=True,
        n_jobs=-1
    )

    # ---------- Pipeline with SMOTEENN for aggressive balancing ----------
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', SMOTEENN(random_state=42)),
        ('stack', stack)
    ])

    # ---------- Hyperparameter tuning ----------
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

    # ---------- Train final model ----------
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ---------- Ranking Predictions ----------
    y_scores = best_model.predict_proba(X_test)[:, 1]
    X_test_df = pd.DataFrame(X_test, columns=top_features)
    X_test_df['score'] = y_scores
    X_test_df['true_label'] = y_test.values
    X_test_df_sorted = X_test_df.sort_values('score', ascending=False).reset_index(drop=True)

    # ---------- Predict top-k bankruptcies ----------
    top_n = round(0.20 * len(X_test_df_sorted))
    X_test_df_sorted['predicted'] = 0
    X_test_df_sorted.loc[:top_n - 1, 'predicted'] = 1

    return 'passed'