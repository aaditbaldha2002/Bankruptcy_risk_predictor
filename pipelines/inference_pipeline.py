import logging
from typing import List
import numpy as np

from zenml.pipelines import pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

@pipeline(enable_cache=True)
def inference_pipeline(data) -> List[float]:
    try:
        # Load classification model
        classification_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        cluster_label = int(classification_model.predict(data)[0])
        logging.info(f"Predicted cluster label: {cluster_label}")

    except Exception as e:
        logging.error(f"[Classification] Failed to predict cluster label: {e}", exc_info=True)
        raise

    try:
        # Dictionary to simulate loading cluster-specific models
        cluster_models = {
            0: LinearRegression(),
            1: LinearRegression(),
            2: LinearRegression(),
            3: LinearRegression(),
            4: LinearRegression()
        }

        if cluster_label not in cluster_models:
            raise ValueError(f"No model found for cluster label: {cluster_label}")

        selected_model = cluster_models[cluster_label]
        prediction = selected_model.predict(data)

        logging.info(f"Prediction for cluster {cluster_label}: {prediction}")
        return prediction.tolist() if isinstance(prediction, np.ndarray) else prediction

    except Exception as e:
        logging.error(f"[Inference] Failed to generate prediction for cluster {cluster_label}: {e}", exc_info=True)
        raise
