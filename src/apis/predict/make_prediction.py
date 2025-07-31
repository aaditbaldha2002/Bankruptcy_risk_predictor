import pandas as pd
from src.apis.predict.constants import FEATURE_NAMES
from src.apis.predict.schemas import BankruptcyPredictionInput
import boto3
from src.apis.predict.utils import api_cluster_predict_data, inference_api_preprocess_data

def make_prediction(payload:BankruptcyPredictionInput):
    data_dict=payload.dict()
    ordered_data = {col: data_dict.get(col, 0) for col in FEATURE_NAMES}
    df = pd.DataFrame([ordered_data])
    processed_df=inference_api_preprocess_data(df)
    cluster_label=api_cluster_predict_data(processed_df)
    final_prediction=api_predict_on_cluster_label(processed_df,cluster_label)