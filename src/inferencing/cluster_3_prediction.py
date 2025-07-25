import logging
import pandas as pd

from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(3)
def cluster_3_prediction(file_path)->int:
    pass