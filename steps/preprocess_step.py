import logging
import pandas as pd
from zenml.steps import step
from src.preprocessing.preprocessing import preprocess_data

@step(enable_cache=False)
def preprocess_step(data_path: str) -> str:
    logging.info('Started preprocess_step...')
    try:
        logging.info(f"[Preprocessing] Starting preprocessing for: {data_path}")
        transformed_data_path = preprocess_data(data_path)
        logging.info(f"[Preprocessing] Preprocessing completed. Transformed data saved at: {transformed_data_path}")
        logging.info('preprocess_step completed')
        return transformed_data_path

    except Exception as e:
        logging.error(f"[Preprocessing] Failed to preprocess data at {data_path}: {e}", exc_info=True)
        raise
