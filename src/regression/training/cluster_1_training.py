
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

@register_trainer(1)
def cluster_1_training(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    X = dataset.drop(["Bankrupt?", "Index"], axis=1)
    y = dataset["Bankrupt?"]

    # --- 2. Train-Test Split (Stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 3. Standardization ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. ADASYN Oversampling ---
    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_scaled, y_train)

    # --- 5. Model and Expanded Hyperparameter Grid ---
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [6, 9, 12, 15],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [10, 20, 26, 30],  # Class imbalance tuning
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.1, 0.3]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

    # --- 6. Fit Grid Search on Resampled Data ---
    grid.fit(X_train_res, y_train_res)

    print("✅ Best Params:", grid.best_params_)

    best_model = grid.best_estimator_

    # --- 7. Predict Probabilities on Test Set ---
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    # --- 8. Threshold Tuning ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    optimal_threshold = 0.65  # Tune based on PR tradeoff
    y_pred_thresh = (y_prob >= optimal_threshold).astype(int)
