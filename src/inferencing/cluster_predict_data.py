import logging
import os

import joblib

def cluster_predict_data(data_path:str)->int:
    try:
        input_data=joblib.load(data_path)
        CLUSTERING_DIR=os.path.join('../../artifacts','clustering')

        min_max_scaler=joblib.load(os.path.join(CLUSTERING_DIR,'min_max_scaler_before_clustering.pkl'))
        df=min_max_scaler.transform(input_data)
        
        MODEL_REGISTRY_DIR=os.path.join('../../model_registry')
        CLASSIFIER_DIR=os.path.join(MODEL_REGISTRY_DIR,'latest_models','classifier')
        classifier_model=joblib.load(os.path.join(CLASSIFIER_DIR,'model.pkl'))

        cluster_label=classifier_model.predict(df)
        return cluster_label
    except Exception as e:
        logging.error(f"Error occurred while predicting cluster of the input data:{e}")
        raise e
