import logging
import os
import joblib
import pandas as pd

from src.inferencing.predict_on_cluster_label import register_inferrer

@register_inferrer(1)
def cluster_1_prediction(file_path:str)->int:
    ARTIFACTS_DIR=os.path.join('artifacts','cluster_1')
    PREPROCESS_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing')
    MODEL_REGISTRY=os.path.join('model_registry','latest_models','cluster_1_regressor')
    scaler=joblib.load(os.path.join(PREPROCESS_DIR,'cluster_1_standard_scaler.pkl'))
    cols_to_drop=joblib.load(os.path.join(PREPROCESS_DIR,'cluster_1_columns_to_drop.pkl'))

    input_data=pd.read_csv(file_path)
    transformed_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

    final_input_data=transformed_input_data.drop(columns=cols_to_drop)

    cluster_1_model=joblib.load(os.path.join(MODEL_REGISTRY,'model.pkl'))

    final_prediction=cluster_1_model.predict(final_input_data)
    logging.info(f"Final prediction by cluster 1 regressor:{final_prediction}")
    return int(final_prediction[0])