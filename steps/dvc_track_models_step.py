import os
import shutil
import subprocess
from typing import List, Tuple
from mlflow.tracking import MlflowClient
from zenml import step
import glob


def run_cmd(command: str):
    """Run shell commands with error capture."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    print(result.stdout)


@step(enable_cache=False)
def dvc_track_models_step(
    artifact_classifier_model_uri: str,
    artifact_regressor_model_uris: List[str],
    classifier_model_uri: str,
    regressor_model_uris: List[str],
    local_path: str = "model_registry/latest_models"
) -> Tuple[str,List[str]]:
    client = MlflowClient()
    local_regressor_paths=[]
    # Cleanup local path
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    # === Classifier Model ===
    classifier_local_path = os.path.join(local_path, "classifier")
    classifier_run_id = classifier_model_uri.split("/")[1]
    classifier_artifact_path = client.download_artifacts(classifier_run_id, "cluster_classification_model")

    print(f"Downloaded classifier model to: {classifier_artifact_path}")

    if os.path.exists(classifier_local_path):
        shutil.rmtree(classifier_local_path)
    shutil.copytree(classifier_artifact_path, classifier_local_path)

    run_cmd(f'dvc add "{classifier_local_path}"')
    local_classifier_path="model_registry/latest_models/classifier/model.pkl"
    # === Regressor Models ===
    for i, reg_uri in enumerate(regressor_model_uris):
        reg_local_path = os.path.join(local_path, f"cluster_{i}_regressor")
        reg_run_id = reg_uri.split("/")[1]
        reg_artifact_path = client.download_artifacts(reg_run_id, f"cluster_{i}_regression_model")

        print(f"Downloaded regressor model {i} to: {reg_artifact_path}")

        if os.path.exists(reg_local_path):
            shutil.rmtree(reg_local_path)
        shutil.copytree(reg_artifact_path, reg_local_path)

        run_cmd(f'dvc add "{reg_local_path}"')
        local_regressor_paths.append(f"model_registry/latest_models/cluster_{i}_regressor/model.pkl")

    # === Git Add .dvc Files ===
    dvc_files = glob.glob(f"{local_path}/*.dvc")
    for dvc_file in dvc_files:
        run_cmd(f'git add "{dvc_file}"')

    # === Git Commit if there are changes ===
    try:
        run_cmd('git diff --cached --quiet || git commit -m "Track updated classifier and regressors in DVC"')
    except RuntimeError as e:
        print("No changes to commit.")

    # === DVC Push to remote ===
    run_cmd('dvc push')
    return local_classifier_path,local_regressor_paths
