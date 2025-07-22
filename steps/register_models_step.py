import logging
from typing import List, Tuple
from mlflow import MlflowClient
from zenml.steps import step

@step
def register_models_step(classifier_model_uri:str,regressor_model_uris:List[str], deployment_decision:bool)->Tuple[str,List[str]]:
    if deployment_decision:
        client = MlflowClient()
        classifier_model_name="cluster_classification_model"
        try:
            # Will only create once
            client.create_registered_model(classifier_model_name)
        except Exception as e:
            logging.error(f"Error in creating registered model for classifier_model_name: {e}")
            raise e

        version = client.create_model_version(name=classifier_model_name, source=classifier_model_uri, run_id=classifier_model_uri.split('/')[1])
        client.transition_model_version_stage(
            name=classifier_model_name,
            version=version.version,
            stage="Staging"
        )
        artifact_classifier_model_uri=f"models:/{classifier_model_name}/{version.version}"

        artifact_regressor_model_uris=[]
        for cluster_id,regressor_model_uri in enumerate(regressor_model_uris):
            regressor_model_name=f"cluster_{cluster_id}_regressor_model"
            try:
                # Will only create once
                client.create_registered_model(regressor_model_name)
            except Exception as e:
                logging.error(f"Error in creating registered model for classifier_model_name: {e}")
                raise e

            version = client.create_model_version(name=regressor_model_name, source=regressor_model_uri, run_id=regressor_model_uri.split('/')[1])
            client.transition_model_version_stage(
                name=regressor_model_name,
                version=version.version,
                stage="Staging"
            )
            artifact_regressor_model_uri=f"models:/{regressor_model_name}/{version.version}"
            artifact_regressor_model_uris.append(artifact_regressor_model_uri)
        return artifact_classifier_model_uri,artifact_regressor_model_uris

    else:
        logging.error("The models cannot be deployed because the deployment decision is not made")
        raise