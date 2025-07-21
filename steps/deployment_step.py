import logging
import subprocess
from typing import Optional
from zenml.steps import step

from src.deployment.deploy_models import deploy_models

@step
def deployment_step(model_uri: str, deploy: bool = True) -> None:
    """
    Serves a model via MLflow if the deployment condition is True.

    Args:
        model_uri (str): MLflow model URI to deploy
        deploy (bool): Flag to deploy or not

    Returns:
        Optional[str]: The URL of the running server if deployed
    """
    logging.info("deployment_step started...")
    deploy_models(deploy,model_uri)
    logging.info("deployment_step completed")
    return