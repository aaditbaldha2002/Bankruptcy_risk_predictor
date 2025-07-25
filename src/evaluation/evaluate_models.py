import logging
from typing import List
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_metrics_from_model_uri(model_uri: str) -> dict:
    run_id = model_uri.split("/")[1]
    client = MlflowClient()
    run_data = client.get_run(run_id).data
    return run_data.metrics

def evaluate_models(
    test_data_path: str,
    classifier_model_uri: str,
    regressor_model_uris: List[str],
    classifier_threshold: float = 0.85,
    average_precision_score_threshold: float = 0.0,
) -> bool:
    logger.info("🔍 Starting evaluation using model URIs...")

    # -----------------------
    # Evaluate Classifier
    # -----------------------
    try:
        logger.info(f"📦 Loading classifier metrics from: {classifier_model_uri}")
        clf_metrics = extract_metrics_from_model_uri(classifier_model_uri)

        clf_accuracy = float(clf_metrics.get("accuracy", -1))
        clf_f1 = float(clf_metrics.get("f1_macro", -1))

        logger.info(f"📊 Classifier Accuracy: {clf_accuracy}, F1 Score: {clf_f1}")

        if clf_accuracy < classifier_threshold:
            logger.error("❌ Classifier accuracy below threshold.")
            return False
        logger.info("✅ Classifier passed accuracy threshold.")
    except Exception:
        logger.exception("❌ Failed to evaluate classifier.")
        return False

    # ---------------------
    # Evaluate Regressors
    # ---------------------
    all_passed = True
    for idx, uri in enumerate(regressor_model_uris):
        try:
            logger.info(f"📦 Loading regressor[{idx}] metrics from: {uri}")
            reg_metrics = extract_metrics_from_model_uri(uri)

            # Ensure correct key is used (replace below if you’re logging actual RMSE as a different name)
            avg_precision_score = float(reg_metrics.get("average_precision_score", 0.5))

            logger.info(f"📊 Regressor[{idx}] Average Precision Score: {avg_precision_score}")

            if avg_precision_score < average_precision_score_threshold:
                logger.error(f"❌ Regressor[{idx}] failed RMSE threshold.")
                all_passed = False
            else:
                logger.info(f"✅ Regressor[{idx}] passed.")
        except Exception:
            logger.exception(f"❌ Failed to evaluate Regressor[{idx}].")
            return False  # Hard fail if any regressor eval fails

    if all_passed:
        logger.info("🚀 All models passed thresholds. Deployment approved.")
        return True
    else:
        logger.warning("⛔ One or more regressors failed. Deployment blocked.")
        return False
