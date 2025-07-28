import logging
from typing import List
from zenml.steps import step

from src.push_to_s3 import promote_models_to_s3

@step(enable_cache=True)
def push_models_to_s3_step(local_classifier_path:str,local_regressor_paths:List[str])->None:
    logging.info("Starting the step for pushing models to s3...")
    try:
        promote_models_to_s3(local_classifier_path,local_regressor_paths)
        logging.info("Models pushed to s3")
    except Exception as e:
        logging.error(f"Error occured while trying to push models to s3: {e}")
        raise e
    return