import logging
from typing import Callable
import pandas as pd

INFERENCE_REGISTRY: dict[int, Callable[[str], int]] = {}
def register_inferrer(cluster_id: int):
    def decorator(func: Callable[[str], int]):
        INFERENCE_REGISTRY[cluster_id] = func
        return func
    return decorator

def predict_on_cluster_label(transformed_file_path:str,cluster_label:int)->int:
    if cluster_label not in INFERENCE_REGISTRY:
        raise ValueError(f"No inferrer registered for cluster {cluster_label}")
    logging.info(f"🧠 Inferencing model for cluster {cluster_label} using {transformed_file_path}")
    return INFERENCE_REGISTRY[cluster_label](transformed_file_path)