import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

@register_trainer(4)
def cluster_4_training(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    X = dataset.drop(columns=['Bankrupt?', 'Index'])
    y = dataset['Bankrupt?'].values  # numpy array for indexing performance

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------
    # XGBoost Model Setup
    # ---------------------------
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
        n_jobs=-1  # utilize all CPU cores
    )

    # ---------------------------
    # Stratified K-Fold CV with SMOTE & Prediction Aggregation
    # ---------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE only on training fold
        X_resampled, y_resampled = SMOTE(random_state=42, n_jobs=-1).fit_resample(X_train, y_train)

        model = clone(base_model)
        model.fit(X_resampled, y_resampled, verbose=False)

        y_probs[test_idx] = model.predict_proba(X_test)[:, 1]

    # ---------------------------
    # Optimize Threshold Based on F2 Score
    # ---------------------------
    precision, recall, thresholds = precision_recall_curve(y, y_probs)
    f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-10)

    # thresholds array length is len(precision) - 1
    best_idx = np.argmax(f2_scores[:-1])
    best_thresh = thresholds[best_idx]

    y_pred = (y_probs >= best_thresh).astype(int)

    return 'passed'