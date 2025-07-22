import os
import shutil
from typing import List
from mlflow.tracking import MlflowClient
from zenml import step

@step
def dvc_track_models_step(
    artifact_classifier_model_uri: str,
    artifact_regressor_model_uris: List[str],
    local_path: str = "model_registry/latest_models"
) -> None:
    client = MlflowClient()

    # Clean up the old model registry path
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    # -------- Classifier Model --------
    classifier_local_path = os.path.join(local_path, "classifier")
    classifier_run_id = artifact_classifier_model_uri.split("/")[1]
    classifier_artifact_path = client.download_artifacts(classifier_run_id, "model")
    shutil.copytree(classifier_artifact_path, classifier_local_path)

    # -------- Regressor Models --------
    for i, reg_uri in enumerate(artifact_regressor_model_uris):
        reg_local_path = os.path.join(local_path, f"regressor_{i}")
        reg_run_id = reg_uri.split("/")[1]
        reg_artifact_path = client.download_artifacts(reg_run_id, "model")
        shutil.copytree(reg_artifact_path, reg_local_path)

    # -------- DVC Tracking --------
    os.system(f'dvc add {local_path}')
    os.system(f'git add {local_path}.dvc')
    os.system('git commit -m "Overwrite latest classifier and regressors from MLflow runs"')
    os.system('dvc push')
