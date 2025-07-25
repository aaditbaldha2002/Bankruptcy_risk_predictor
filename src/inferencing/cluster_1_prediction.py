import logging
import pandas as pd

from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(1)
def cluster_1_prediction(file_path:str)->int:
    pass