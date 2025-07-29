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
    MODEL_DIR=os.path.join(ARTIFACTS_DIR,'model')

    input_data=pd.read_csv(file_path)
    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'standard_scaler.pkl'))
    transformed_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

    cols_to_drop=joblib.load(os.path.join(PREPROCESS_DIR,'cols_to_drop_before_pca.pkl'))
    pca_input_data=transformed_input_data.drop(columns=cols_to_drop)

    dropped_cols=joblib.load(os.path.join(PCA_DIR,'columns_to_drop.pkl'))
    pca_pairs_df=joblib.load(os.path.join(PCA_DIR,'pca_pairs_used.pkl'))
    pca_models=joblib.load(os.path.join(PCA_DIR,'fitted_pca_models.pkl'))
    best_threshold=joblib.load(os.path.join(MODEL_DIR,'model_threshold.pkl'))

    pca_input_data=pca_input_data.drop(columns=dropped_cols)
    pca_transformed_data=inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
    
    cluster_0_model=joblib.load(os.path.join(MODEL_REGISTRY_DIR,'cluster_0_regressor','model.pkl'))
    final_prediction=cluster_0_model.predict(pca_transformed_data)
    if final_prediction < best_threshold:
        final_prediction=0
    else:
        final_prediction=1
    logging.info(f"Final prediction by the cluster 0 regressor:{final_prediction}")
    return final_prediction