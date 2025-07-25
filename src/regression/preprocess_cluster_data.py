import logging
from typing import Callable

logger = logging.getLogger(__name__)
PREPROCESSING_REGISTRY: dict[int, Callable[[str], str]] = {}

def register_preprocessor(cluster_id: int):
    def decorator(func: Callable[[str], str]):
        PREPROCESSING_REGISTRY[cluster_id] = func
        return func
    return decorator

def preprocess_cluster_data(cluster_id:int,data_path:str)->str:
    """
    Dynamically dispatches to the appropriate preprocessing function based on cluster_id.
    """
    logger.info(f"🔧 Preprocessing for cluster {cluster_id} at {data_path}")
    if cluster_id not in PREPROCESSING_REGISTRY:
        raise ValueError(f"No preprocessing registered for cluster {cluster_id}")
    return PREPROCESSING_REGISTRY[cluster_id](data_path)