import logging
import os
import joblib
import pandas as pd

from src.inferencing.inference_pca_transform import inference_pca_transform
from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(4)
def cluster_4_prediction(file_path)->int:
    ARTIFACTS_DIR=os.path.join('artifacts','cluster_4')
    PREPROCESS_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing')
    PCA_DIR=os.path.join(PREPROCESS_DIR,'pca')
    MODEL_REGISTRY=os.path.join('model_registry','latest_models','cluster_4_regressor')

    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'scaler.pkl'))
    input_data = pd.read_csv(file_path)
    dropped_cols=joblib.load(PCA_DIR,'columns_to_drop.pkl')
    pca_pairs_df=joblib.load(PCA_DIR,'pca_pairs_used.pkl')
    pca_models=joblib.load(PCA_DIR,'fitted_pca_models.pkl')

    input_data=joblib.load(file_path)
    scaled_input_data=scaler.transform(input_data)
    pca_input_data=scaled_input_data.drop(columns=dropped_cols)
    pca_transformed_data=inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
    
    cluster_4_model=joblib.load(os.path.join(MODEL_REGISTRY,'model.pkl'))
    final_prediction=cluster_4_model.predict(pca_transformed_data)

    return final_prediction