import logging
import pandas as pd

from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(4)
def cluster_4_prediction(file_path)->int:
    pass