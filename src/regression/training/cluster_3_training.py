import logging
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from src.regression.train_model_for_cluster import register_trainer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

@register_trainer(3)
def cluster_3_training(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    X=dataset.drop(columns=['Bankrupt?']).values
    y=dataset['Bankrupt?'].values

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Pipeline: SMOTETomek → Scaler → XGBoost
    pipeline = ImbPipeline([
        ('resample', SMOTETomek(random_state=42)),
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=1,  # Let sampling handle imbalance
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Hyperparameter grid
    param_distributions = {
        'clf__n_estimators': randint(300, 600),
        'clf__max_depth': randint(3, 10),
        'clf__learning_rate': uniform(0.01, 0.3),
        'clf__min_child_weight': randint(1, 6),
        'clf__subsample': uniform(0.6, 0.4),
        'clf__colsample_bytree': uniform(0.6, 0.4)
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Search CV
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

    # Fit model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict probabilities
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Find optimal threshold using F2-score (recall-oriented)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-6)
    best_threshold = thresholds[np.argmax(f2_scores)]

    # Final prediction
    y_pred = (y_proba >= best_threshold).astype(int)

    X_resampled, y_resampled = SMOTETomek(random_state=42).fit_resample(X_train, y_train)

    return 'passed'