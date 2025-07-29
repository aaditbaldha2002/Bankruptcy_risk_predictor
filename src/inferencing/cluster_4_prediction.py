import logging
import os
import joblib
import numpy as np
import pandas as pd

from src.inferencing.inference_pca_transform import inference_pca_transform
from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(4)
def cluster_4_prediction(file_path:str)->int:
    ARTIFACTS_DIR=os.path.join('artifacts','cluster_4')
    PREPROCESS_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing')
    PCA_DIR=os.path.join(PREPROCESS_DIR,'pca')
    MODEL_REGISTRY=os.path.join('model_registry','latest_models','cluster_4_regressor')

    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'cluster_4_standard_scaler.pkl'))
    input_data = pd.read_csv(file_path)
    dropped_cols=joblib.load(os.path.join(PCA_DIR,'cluster_4_columns_to_drop.pkl'))
    pca_pairs_df=joblib.load(os.path.join(PCA_DIR,'cluster_4_pca_pairs_used.pkl'))
    pca_models=joblib.load(os.path.join(PCA_DIR,'cluster_4_fitted_pca_models.pkl'))

    scaled_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)
    pca_input_data=scaled_input_data.drop(columns=dropped_cols)
    pca_transformed_data=inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
    
    cluster_4_model=joblib.load(os.path.join(MODEL_REGISTRY,'model.pkl'))
    final_prediction=cluster_4_model.predict(pca_transformed_data)

    return int(final_prediction[0])