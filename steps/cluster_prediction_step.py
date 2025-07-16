import logging
from typing import List, Tuple

import pandas as pd
from zenml.steps import step

from src.clustering.clustering import clustering



logger = logging.getLogger(__name__)

@step(enable_cache=True)
def cluster_prediction_step(data_path: str) -> Tuple[str,List[str]]:
    try:
        cluster_labeled_file_path,cluster_file_paths = clustering(data_path)
        logger.info(f"Clustering completed successfully, generated files: {cluster_labeled_file_path},{cluster_file_paths}")
        return cluster_labeled_file_path,cluster_file_paths
    except Exception as e:
        logger.error(f"Error in cluster_prediction_step: {e}")
        raise
