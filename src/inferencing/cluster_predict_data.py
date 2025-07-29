import logging
import os

import joblib
import pandas as pd
import numpy as np

def cluster_predict_data(data_path:str)->int:
    try:
        input_data=pd.read_csv(data_path)
        CLUSTERING_DIR=os.path.join('artifacts','clustering')

        min_max_scaler=joblib.load(os.path.join(CLUSTERING_DIR,'min_max_scaler_before_clustering.pkl'))
        df = pd.DataFrame(
            min_max_scaler.transform(input_data),
            columns=input_data.columns,
            index=input_data.index
        )
        
        MODEL_REGISTRY_DIR=os.path.join('model_registry')
        CLASSIFIER_DIR=os.path.join(MODEL_REGISTRY_DIR,'latest_models','classifier')
        classifier_model=joblib.load(os.path.join(CLASSIFIER_DIR,'model.pkl'))
        
        top_28_features=joblib.load(os.path.join(CLUSTERING_DIR,'top_28_features.pkl'))
        df=df[top_28_features]
        cluster_label=classifier_model.predict(df)
        return int(cluster_label[0])
    except Exception as e:
        logging.error(f"Error occurred while predicting cluster of the input data:{e}")
        raise e
