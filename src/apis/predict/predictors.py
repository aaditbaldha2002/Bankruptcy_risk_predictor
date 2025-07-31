import logging
import os
import joblib
import pandas as pd
from src.apis.predict.constants import S3_BUCKET_NAME
from src.apis.predict.utils import api_inference_pca_transform, api_register_inferrer, parse_artifact_mapping, parse_model_paths
import boto3

@api_register_inferrer(0)
def api_cluster_0_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the api_cluster_0_prediction function...")
        s3=boto3.client('s3')
        MODEL_DIR_OBJECT=S3_BUCKET_NAME[:-9]

        input_data=df
        scaler_hash=parse_artifact_mapping('cluster_0_standard_scaler.pkl')
        scaler=s3.get_object(Bucket=S3_BUCKET_NAME,key=scaler_hash)

        transformed_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

        cols_to_drop_hash=parse_artifact_mapping('cluster_0_cols_to_drop_before_pca.pkl')
        cols_to_drop=s3.get_object(Bucket=S3_BUCKET_NAME,key=cols_to_drop_hash)
        pca_input_data=transformed_input_data.drop(columns=cols_to_drop)

        dropped_cols_hash=parse_artifact_mapping('cluster_0_columns_to_drop.pkl')
        dropped_cols=s3.get_object(Bucket=S3_BUCKET_NAME,key=dropped_cols_hash)
        pca_pairs_df_hash=parse_artifact_mapping('cluster_0_pca_pairs_used.pkl')
        pca_pairs_df=s3.get_object(Bucket=S3_BUCKET_NAME,key=pca_pairs_df_hash)
        pca_models_hash=parse_artifact_mapping('cluster_0_fitted_pca_models.pkl')
        pca_models=s3.get_object(Bucket=S3_BUCKET_NAME,key=pca_models_hash)
        best_threshold_hash=parse_artifact_mapping('cluster_0_model_threshold.pkl')
        best_threshold=s3.get_object(Bucket=S3_BUCKET_NAME,key=best_threshold_hash)

        pca_input_data=pca_input_data.drop(columns=dropped_cols)
        pca_transformed_data=api_inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
        
        cluster_0_model_path=parse_model_paths('cluster_0_regressor')
        cluster_0_model=s3.get_object(Bucket=MODEL_DIR_OBJECT,key=cluster_0_model_path)
        final_prediction=cluster_0_model.predict(pca_transformed_data)
        if final_prediction < best_threshold:
            final_prediction=0
        else:
            final_prediction=1
        logging.info(f"Final prediction by the cluster 0 regressor:{final_prediction}")
        return final_prediction
    except Exception as e:
        logging.error(f"Error occured in the api_cluster_0_prediction function:{e}")