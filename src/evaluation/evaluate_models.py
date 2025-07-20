import logging
from typing import List
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_metrics_from_model_uri(model_uri: str) -> dict:
    """
    Extracts metrics from an MLflow run using the model URI.
    Args:
        model_uri (str): e.g., runs:/<run_id>/model_artifact_path

    Returns:
        dict: Metrics dictionary
    """
    run_id = model_uri.split("/")[1]
    client = MlflowClient()
    run_data = client.get_run(run_id).data
    return run_data.metrics


def evaluate_models(
    test_data_path: str,
    classifier_model_uri: str,
    regressor_model_uris: List[str],
    classifier_threshold: float = 0.0,
    regressor_rmse_threshold: float = 0,
    regressor_r2_threshold: float = 0
) -> bool:
    logger.info("🔍 Starting evaluation using model URIs...")

    # -----------------------
    # Evaluate Classification
    # -----------------------
    try:
        logger.info(f"📦 Loading classifier metrics from: {classifier_model_uri}")
        clf_metrics = extract_metrics_from_model_uri(classifier_model_uri)
        clf_accuracy = float(clf_metrics.get("accuracy", 0.0))
        clf_f1 = float(clf_metrics.get("f1_macro", 0.0))

        logger.info(f"📊 Classifier Accuracy: {clf_accuracy}, F1 Score: {clf_f1}")

        if clf_accuracy < classifier_threshold:
            logger.warning("❌ Classifier failed threshold.")
            return False
        logger.info("✅ Classifier passed.")
    except Exception as e:
        logger.exception("❌ Failed to evaluate classifier.")
        return False

    # ---------------------
    # Evaluate Regressors
    # ---------------------
    passed_regressors = 0
    for idx, uri in enumerate(regressor_model_uris):
        try:
            logger.info(f"📦 Loading regressor metrics from: {uri}")
            reg_metrics = extract_metrics_from_model_uri(uri)

            rmse = float(reg_metrics.get("average_precision_score", 1e9))  # Default to large number

            if rmse <= regressor_rmse_threshold:
                passed_regressors += 1
                logger.info(f"✅ Regressor[{idx}] passed.")
            else:
                logger.warning(f"❌ Regressor[{idx}] failed.")
        except Exception as e:
            logger.exception(f"❌ Failed to evaluate Regressor[{idx}].")
            continue

    if passed_regressors == len(regressor_model_uris):
        logger.info("🚀 All regressors passed. Deployment approved.")
        return True
    else:
        logger.warning(f"⛔ Only {passed_regressors}/{len(regressor_model_uris)} regressors passed.")
        return False
