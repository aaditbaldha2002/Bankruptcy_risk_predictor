import logging
from typing import List, Tuple
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from zenml.steps import step

@step
def register_models_step(classifier_model_uri: str, regressor_model_uris: List[str], deployment_decision: bool) -> Tuple[str, List[str],str,List[str]]:
    if not deployment_decision:
        logging.error("The models cannot be deployed because the deployment decision is not made.")
        raise ValueError("Deployment decision was False.")

    client = MlflowClient()

    def recreate_registered_model(model_name: str):
        try:
            client.get_registered_model(model_name)
            logging.warning(f"Model '{model_name}' already exists. Deleting it.")
            client.delete_registered_model(name=model_name)
        except Exception as e:
            logging.info(f"Model '{model_name}' does not exist. Proceeding to create.")
            
        client.create_registered_model(model_name)
        logging.info(f"Model '{model_name}' registered successfully.")

    # --- Register Classifier Model ---
    classifier_model_name = "cluster_classification_model"
    recreate_registered_model(classifier_model_name)
    classifier_version = client.create_model_version(
        name=classifier_model_name,
        source=classifier_model_uri,
        run_id=classifier_model_uri.split("/")[1]
    )
    client.transition_model_version_stage(
        name=classifier_model_name,
        version=classifier_version.version,
        stage="Staging"
    )
    artifact_classifier_model_uri = f"models:/{classifier_model_name}/{classifier_version.version}"

    # --- Register Regressor Models ---
    artifact_regressor_model_uris = []
    for cluster_id, regressor_model_uri in enumerate(regressor_model_uris):
        regressor_model_name = f"cluster_{cluster_id}_regression_model"
        recreate_registered_model(regressor_model_name)
        regressor_version = client.create_model_version(
            name=regressor_model_name,
            source=regressor_model_uri,
            run_id=regressor_model_uri.split("/")[1]
        )
        client.transition_model_version_stage(
            name=regressor_model_name,
            version=regressor_version.version,
            stage="Staging"
        )
        artifact_regressor_model_uris.append(f"models:/{regressor_model_name}/{regressor_version.version}")

    return artifact_classifier_model_uri, artifact_regressor_model_uris,classifier_model_uri,regressor_model_uris
