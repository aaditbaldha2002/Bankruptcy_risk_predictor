import logging
from typing import List

import pandas as pd
from zenml.steps import step

from src.inferencing.inference_preprocess_data import inference_preprocess_data

@step(enable_cache=False)
def inference_preprocess_data_step()->str:
    try:
        logging.info('inference_preprocess_data_step step starting...')
        transformed_data_file_path=inference_preprocess_data()
        logging.info('inference_preprocess_data_step step completed')
        return transformed_data_file_path
    except Exception as e:
        logging.error(f"Error occurred in inference_preprocess_data_step: {e}")
        raise e