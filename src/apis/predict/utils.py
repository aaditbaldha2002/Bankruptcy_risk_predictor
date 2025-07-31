import io
from typing import Callable
import pandas as pd
import os
import joblib
import numpy
import logging
import json
import boto3
import numpy as np
from src.apis.predict.constants import INFERENCE_REGISTRY, MODEL_PATHS, S3_BUCKET_NAME

def load_pickle_from_s3(Bucket:str, Key:str):
    print(f"Loading pickle from S3 bucket: {Bucket}, key: {Key}")
    logging.info(f"Loading pickle from S3 bucket: {Bucket}, key: {Key}")

    s3=boto3.client('s3')
    obj = s3.get_object(Bucket=Bucket, Key=Key)
    return joblib.load(io.BytesIO(obj['Body'].read()))

def parse_artifact_mapping(required_file_name:str)->str:
    with open('dvc_artifact_manifest.json', 'r') as f:
            hashMap = json.load(f)    
    return hashMap[required_file_name]

def parse_model_paths(required_model_name:str)->str:
    mapper=MODEL_PATHS
    return mapper[required_model_name]

def inference_api_preprocess_data(df:pd.DataFrame) -> str:
    df.columns=df.columns.str.strip()
    s3=boto3.client('s3')
    scaler_file_name_hash=parse_artifact_mapping('preprocess_scaler.pkl')
    scaler = load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2], Key=scaler_file_name_hash)
    print("error starts from next line")

    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Drop columns before PCA
    preprocess_cols_to_drop_hash=parse_artifact_mapping('preprocess_columns_to_drop_before_pca.pkl')
    columns_to_drop=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=preprocess_cols_to_drop_hash)
    df.drop(columns=columns_to_drop, inplace=True)

    # Load PCA components
    dropped_cols_hash=parse_artifact_mapping('preprocess_columns_to_drop.pkl')
    dropped_cols=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=dropped_cols_hash)
    pca_pairs_df_hash=parse_artifact_mapping('preprocess_pca_pairs_used.pkl')
    pca_pairs_df=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_pairs_df_hash)
    pca_models_hash=parse_artifact_mapping('preprocess_fitted_pca_models.pkl')
    pca_models=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=pca_models_hash)

    df.drop(columns=dropped_cols, inplace=True)

    new_cols = []
    for _, row in pca_pairs_df.iterrows():
        f1, f2 = row['Feature_1'], row['Feature_2']
        if f1 not in df.columns or f2 not in df.columns:
            continue

        subset = df[[f1, f2]].dropna()
        if subset.empty:
            continue

        adjusted = subset.values - subset.values.mean(axis=0)
        key = f"{sorted([f1, f2])[0]}__{sorted([f1, f2])[1]}"
        pca = pca_models.get(key)

        if pca is None:
            continue

        new_col = f"PCA_{f1}_{f2}"
        df[new_col] = np.nan
        df.loc[subset.index, new_col] = pca.transform(adjusted).flatten()
        df.drop(columns=[f1, f2], inplace=True)
        new_cols.append(new_col)

    # Drop post-PCA noise features
    cols_to_drop_after_pca_hash=parse_artifact_mapping('preprocess_columns_to_drop_after_pca.pkl')
    cols_to_drop_after_pca=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=cols_to_drop_after_pca_hash)
    df.drop(columns=cols_to_drop_after_pca, inplace=True)

    # Load adaptive transformers
    transformers_hash=parse_artifact_mapping('preprocess_adaptive_transformers.pkl')
    try:
        transformers=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=transformers_hash)
        logging.info(f"Loaded adaptive transformers from S3")
    except Exception as e:
        logging.exception(f"Could not load transformers: {e}")
        raise

    # Ensure required features exist
    missing_features = [f for f in transformers.keys() if f not in df.columns]
    if missing_features:
        raise ValueError(f"Inference data is missing required features: {missing_features}")

    # Transform each feature
    transformed_data = {}
    for feature, transformer in transformers.items():
        try:
            x = df[feature].values.reshape(-1, 1)
            transformed_feature = transformer.transform(x)
            transformed_data[feature] = transformed_feature.flatten()
        except Exception as e:
            logging.exception(f"Failed to transform feature '{feature}': {e}")
            raise RuntimeError(f"Transformation failed for feature '{feature}': {e}")

    transformed_df = pd.DataFrame(transformed_data)
    return transformed_df

def api_cluster_predict_data(df:pd.DataFrame)->int:
    try:
        logging.info("api_cluster_predict_data function started...")
        transformed_df=df
        s3=boto3.client('s3')
        MODEL_S3_DIR=S3_BUCKET_NAME.split('/')[2]
        min_max_scaler_hash=parse_artifact_mapping('min_max_scaler_before_clustering.pkl')
        min_max_scaler=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=min_max_scaler_hash)
        print("error is after this line")

        print("Min Max scaling the dataset...")
        logging.info("Min Max scaling the dataset...")
        df = pd.DataFrame(
            min_max_scaler.transform(transformed_df),
            columns=transformed_df.columns,
            index=transformed_df.index
        )
        print("Min Max scaling completed")
        logging.info("Min max scaling completed.")

        print("Classifier getting loaded")
        classifier_model_s3_path=parse_model_paths('classifier')
        classifier_model=load_pickle_from_s3(Bucket=MODEL_S3_DIR,Key=classifier_model_s3_path)
        print("CLassifier loaded")

        print("Dropping less imp columns")
        top_28_features_hash=parse_artifact_mapping('classifier_top_28_features.pkl')
        top_28_features=load_pickle_from_s3(Bucket=S3_BUCKET_NAME.split('/')[2],Key=top_28_features_hash)
        logging.info("dropping trivial feature columns for cluster label prediction...")
        df=df[top_28_features]
        print("trivial feature column dropped.")
        logging.info("trivial feature column dropped.")
        print("Predicting the cluster label...")
        logging.info("Predicting the cluster label...")
        cluster_label=classifier_model.predict(df)
        logging.info("cluster label predicted successfully")
        return int(cluster_label[0])
    
    except Exception as e:
        logging.error(f"Error occurred while predicting cluster of the input data:{e}")
        raise e



def api_register_inferrer(cluster_id: int):
    def decorator(func: Callable[[pd.DataFrame], int]):
        INFERENCE_REGISTRY[cluster_id] = func
        return func
    return decorator

def api_predict_on_cluster_label(df:pd.DataFrame,cluster_label:int)->int:
    try:
        logging.info("Prediction based on cluster label predicted earlier started...")
        if cluster_label not in INFERENCE_REGISTRY:
            raise ValueError(f"No inferrer registered for cluster {cluster_label}")
        logging.info(f"🧠 Inferencing model for cluster {cluster_label} using provided dataframe")
        logging.info("Final prediction completed.")
        return INFERENCE_REGISTRY[cluster_label](df)
    
    except Exception as e:
        logging.error(f"Error occurred in the api_predict_on_cluster_label function:{e}")
        raise e

def api_inference_pca_transform(pca_pairs_df:pd.DataFrame,df:pd.DataFrame,pca_models)->pd.DataFrame:
    new_cols=[]
    for _, row in pca_pairs_df.iterrows():
        f1, f2 = row['Feature_1'], row['Feature_2']
        if f1 not in df.columns or f2 not in df.columns:
            continue

        subset = df[[f1, f2]].dropna()
        if subset.empty:
            continue

        data = subset.values - subset.values.mean(axis=0)
        key = f"{sorted([f1, f2])[0]}__{sorted([f1, f2])[1]}"
        pca = pca_models.get(key)

        if pca is None:
            continue

        new_col = f"PCA_{f1}_{f2}"
        df[new_col] = np.nan
        df.loc[subset.index, new_col] = pca.transform(data).flatten()
        df.drop(columns=[f1, f2], inplace=True)
        new_cols.append(new_col)
    return df

from src.apis.predict.predictors import api_cluster_0_prediction
from src.apis.predict.predictors import api_cluster_1_prediction
from src.apis.predict.predictors import api_cluster_2_prediction
from src.apis.predict.predictors import api_cluster_3_prediction
from src.apis.predict.predictors import api_cluster_4_prediction
