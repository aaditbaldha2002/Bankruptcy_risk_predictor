import logging
from typing import List

from src.inferencing.inference_preprocess_data import inference_preprocess_data

def inference_preprocess_data_step(data:List[float])->None:
    logging.info('inference_preprocess_data_step step starting...')
    inference_preprocess_data(data)
    logging.info('inference_preprocess_data_step step completed')
    pass