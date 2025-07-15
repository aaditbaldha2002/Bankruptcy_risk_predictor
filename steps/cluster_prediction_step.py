import logging
from typing import List

import pandas as pd
from zenml.steps import step

from clustering.clustering import clustering

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def cluster_prediction_step(data_path: str) -> List[str]:
    try:
        file_paths = clustering(data_path)
        logger.info(f"Clustering completed successfully, generated files: {file_paths}")
        return file_paths
    except Exception as e:
        logger.error(f"Error in cluster_prediction_step: {e}")
        raise
