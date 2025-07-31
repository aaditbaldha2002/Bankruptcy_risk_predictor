import logging
import os
import joblib
import pandas as pd
from src.apis.predict.constants import S3_BUCKET_NAME
from src.apis.predict.utils import api_inference_pca_transform, api_register_inferrer, load_pickle_from_s3, parse_artifact_mapping, parse_model_paths
import boto3

@api_register_inferrer(0)
def api_cluster_0_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the api_cluster_0_prediction function...")
        s3=boto3.client('s3')
        MODEL_OBJECT=S3_BUCKET_NAME.split('/')[2]

        input_data=df
        scaler_hash=parse_artifact_mapping('cluster_0_standard_scaler.pkl')
        scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=scaler_hash)

        transformed_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

        cols_to_drop_hash=parse_artifact_mapping('cluster_0_cols_to_drop_before_pca.pkl')
        cols_to_drop=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=cols_to_drop_hash)
        pca_input_data=transformed_input_data.drop(columns=cols_to_drop)

        dropped_cols_hash=parse_artifact_mapping('cluster_0_columns_to_drop.pkl')
        dropped_cols=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=dropped_cols_hash)
        pca_pairs_df_hash=parse_artifact_mapping('cluster_0_pca_pairs_used.pkl')
        pca_pairs_df=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=pca_pairs_df_hash)
        pca_models_hash=parse_artifact_mapping('cluster_0_fitted_pca_models.pkl')
        pca_models=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=pca_models_hash)
        best_threshold_hash=parse_artifact_mapping('cluster_0_model_threshold.pkl')
        best_threshold=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],key=best_threshold_hash)

        pca_input_data=pca_input_data.drop(columns=dropped_cols)
        pca_transformed_data=api_inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
        
        cluster_0_model_path=parse_model_paths('cluster_0_regressor')
        cluster_0_model=load_pickle_from_s3(Bucket=MODEL_OBJECT,key=cluster_0_model_path)
        final_prediction=cluster_0_model.predict(pca_transformed_data)
        if final_prediction < best_threshold:
            final_prediction=0
        else:
            final_prediction=1
        logging.info(f"Final prediction by the cluster 0 regressor:{final_prediction}")
        return final_prediction
    except Exception as e:
        logging.error(f"Error occured in the api_cluster_0_prediction function:{e}")

@api_register_inferrer(1)
def api_cluster_1_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the api_cluster_1_prediction function...")
        s3=boto3.client('s3')

        scaler_hash=parse_artifact_mapping('cluster_1_standard_scaler.pkl')
        scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=scaler_hash)
        cols_to_drop_hash=parse_artifact_mapping('cluster_1_columns_to_drop.pkl')
        cols_to_drop=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=cols_to_drop_hash)

        input_data=df
        transformed_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

        final_input_data=transformed_input_data.drop(columns=cols_to_drop)

        MODEL_BUCKET=S3_BUCKET_NAME.split('/')[2]
        cluster_1_model_path=parse_model_paths('cluster_1_regressor')
        cluster_1_model=load_pickle_from_s3(Bucket=MODEL_BUCKET,Key=cluster_1_model_path)

        final_prediction=cluster_1_model.predict(final_input_data)
        logging.info(f"Final prediction by cluster 1 regressor:{final_prediction}")
        return int(final_prediction[0])
    except Exception as e:
        logging.info(f"Error occurred in function api_cluster_1_prediction function:{e}")

@api_register_inferrer(2)
def api_cluster_2_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the api_cluster_2_prediction function...")
        s3=boto3.client('s3')
        MODEL_BUCKET=S3_BUCKET_NAME.split('/')[2]

        scaler_hash=parse_artifact_mapping('cluster_2_standard_scaler.pkl')
        scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=scaler_hash)
        cols_to_drop_before_pca_hash=parse_artifact_mapping('cluster_2_columns_to_drop.pkl')
        cols_to_drop_before_pca=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=cols_to_drop_before_pca_hash)
        dropped_cols_hash=parse_artifact_mapping('cluster_2_columns_to_drop.pkl')
        dropped_cols=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=dropped_cols_hash)
        pca_pairs_df_hash=parse_artifact_mapping('cluster_2_pca_pairs_used.pkl')
        pca_pairs_df=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_pairs_df_hash)
        pca_models_hash=parse_artifact_mapping('cluster_2_fitted_pca_models.pkl')
        pca_models=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_models_hash)
        cluster_2_model_path=parse_model_paths('cluster_2_regressor')
        cluster_2_model=load_pickle_from_s3(Bucket=MODEL_BUCKET,Key=cluster_2_model_path)
        cols_to_retain_after_pca_hash=parse_artifact_mapping('cluster_2_cols_to_retain_after_pca.pkl')
        cols_to_retain_after_pca=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=cols_to_retain_after_pca_hash)
        input_data = df

        scaled_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)
        pca_input_data=scaled_input_data.drop(columns=cols_to_drop_before_pca)
        pca_input_data=pca_input_data.drop(columns=dropped_cols)
        pca_transformed_input_data=api_inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
        pca_transformed_input_data=pca_transformed_input_data[cols_to_retain_after_pca]
        final_prediction=cluster_2_model.predict(pca_transformed_input_data)
        logging.info(f"Final prediction done by cluster 2 regressor:{final_prediction}")
        return int(final_prediction[0])
    except Exception as e:
        logging.error(f"Error occurred in the api_cluster_2_prediction function:{e}")
        raise e
    
@api_register_inferrer(3)
def api_cluster_3_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the api_cluster_3_prediction function...")
        s3=boto3.client('s3')
        MODEL_BUCKET=S3_BUCKET_NAME.split('/')[2]

        scaler_hash=parse_artifact_mapping('cluster_3_standard_scaler.pkl')
        scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=scaler_hash)
        cols_to_drop_before_pca_hash=parse_artifact_mapping('cluster_3_cols_to_drop_before_pca.pkl')
        cols_to_drop_before_pca=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=cols_to_drop_before_pca_hash)
        dropped_cols_hash=parse_artifact_mapping('cluster_3_columns_to_drop.pkl')
        dropped_cols=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=dropped_cols_hash)
        pca_pairs_df_hash=parse_artifact_mapping('cluster_3_pca_pairs_used.pkl')
        pca_pairs_df=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_pairs_df_hash)
        pca_models_hash=parse_artifact_mapping('cluster_3_fitted_pca_models.pkl')
        pca_models=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_models_hash)
        cluster_3_model_path=parse_model_paths('cluster_3_regressor')
        cluster_3_model=load_pickle_from_s3(Bucket=MODEL_BUCKET,Key=cluster_3_model_path)

        input_data=df
        input_data=input_data.drop(columns=cols_to_drop_before_pca)
        scaled_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)

        pca_input_data=scaled_input_data.drop(columns=dropped_cols)
        pca_transformed_data=api_inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
        
        final_prediction=cluster_3_model.predict(pca_transformed_data)
        
        logging.info(f"Final Prediction made by cluster 3 regressor: {final_prediction}")

        return int(final_prediction[0])
    except Exception as e:
        logging.info(f"Error occurred in the api_cluster_3_prediction function:{e}")

@api_register_inferrer(4)
def api_cluster_4_prediction(df:pd.DataFrame)->int:
    try:
        logging.info("Starting the function api_cluster_4_prediction...")
        s3=boto3.client('s3')
        MODEL_BUCKET=S3_BUCKET_NAME.split('/')[2]

        input_data = df
        scaler_hash=parse_artifact_mapping('cluster_4_standard_scaler.pkl')
        scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=scaler_hash)
        dropped_cols_hash=parse_artifact_mapping('cluster_4_columns_to_drop.pkl')
        dropped_cols=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=dropped_cols_hash)
        pca_pairs_df_hash=parse_artifact_mapping('cluster_4_pca_pairs_used.pkl')
        pca_pairs_df=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_pairs_df_hash)
        pca_models_hash=parse_artifact_mapping('cluster_4_fitted_pca_models.pkl')
        pca_models=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_models_hash)
        cluster_4_model_path=parse_model_paths('cluster_4_regressor')
        cluster_4_model=load_pickle_from_s3(Bucket=MODEL_BUCKET,Key=cluster_4_model_path)

        scaled_input_data=pd.DataFrame(scaler.transform(input_data),columns=input_data.columns)
        pca_input_data=scaled_input_data.drop(columns=dropped_cols)
        pca_transformed_data=api_inference_pca_transform(pca_pairs_df,pca_input_data,pca_models)
        
        final_prediction=cluster_4_model.predict(pca_transformed_data)
        logging.info("Final prediction by the cluster 4 regressor:{final_prediction}")
        return int(final_prediction[0])
    except Exception as e:
        logging.error(f"Error in function api_cluster_4_prediction:{e}")
        raise e