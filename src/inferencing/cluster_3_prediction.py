import logging
import os
import joblib
import pandas as pd

from src.inferencing.inference_pca_transform import inference_pca_transform
from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(3)
def cluster_3_prediction(file_path)->int:
    ARTIFACTS_DIR=os.path.join('artifacts','cluster_3')
    PREPROCESS_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing')
    PCA_DIR=os.path.join(PREPROCESS_DIR,'pca')
    MODEL_REGISTRY=os.path.join('model_registry','latest_models','cluster_3_regressor')

    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'cluster_3_standard_scaler.pkl'))
    cols_to_drop_before_pca=joblib.load(os.path.join(PREPROCESS_DIR,'cluster_3_cols_to_drop_before_pca.pkl'))
    input_data = pd.read_csv(file_path)
    dropped_cols=joblib.load(os.path.join(PCA_DIR,'cluster_3_columns_to_drop.pkl'))
    pca_pairs_df=joblib.load(os.path.join(PCA_DIR,'cluster_3_pca_pairs_used.pkl'))
    pca_models=joblib.load(os.path.join(PCA_DIR,'cluster_3_fitted_pca_models.pkl'))

    input_data=pd.read_csv(file_path)
    input_data=input_data.drop(columns=cols_to_drop_before_pca)
    scaled_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

    pca_input_data=scaled_input_data.drop(columns=dropped_cols)
    pca_transformed_data=inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
    
    cluster_3_model=joblib.load(os.path.join(MODEL_REGISTRY,'model.pkl'))
    final_prediction=cluster_3_model.predict(pca_transformed_data)
    
    logging.info(f"Final Prediction made by cluster 3 regressor: {final_prediction}")

    return int(final_prediction[0])