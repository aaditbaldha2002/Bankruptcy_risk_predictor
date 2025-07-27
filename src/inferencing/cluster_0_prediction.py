import logging
import os
import joblib
import pandas as pd

from src.inferencing.inference_pca_transform import inference_pca_transform
from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(0)
def cluster_0_prediction(file_path:str)->int:
    ARTIFACTS_DIR=os.path.join('artifacts','cluster_0')
    PREPROCESS_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing')
    PCA_DIR=os.path.join(PREPROCESS_DIR,'pca')
    MODEL_REGISTRY_DIR=os.path.join('model_registry','latest_models')

    input_data=pd.read_csv(file_path)
    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'standard_scaler.pkl'))
    transformed_input_data=scaler.transform(input_data)

    cols_to_drop=joblib.load(PREPROCESS_DIR,'cols_to_drop_before_pca.pkl')
    pca_input_data=transformed_input_data.drop(columns=cols_to_drop)

    dropped_cols=joblib.load(PCA_DIR,'columns_to_drop.pkl')
    pca_pairs_df=joblib.load(PCA_DIR,'pca_pairs_used.pkl')
    pca_models=joblib.load(PCA_DIR,'fitted_pca_models.pkl')

    pca_input_data=pca_input_data.drop(columns=[col for col in dropped_cols if col in pca_input_data.columns], inplace=True)
    pca_transformed_data=inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
    
    cluster_0_model=joblib.load(os.path.join(MODEL_REGISTRY_DIR,'cluster_0_regressor','model.pkl'))
    final_prediction=cluster_0_model.predict(pca_transformed_data)
    return final_prediction