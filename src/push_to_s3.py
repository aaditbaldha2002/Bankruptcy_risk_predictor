import logging
import os
import subprocess
from typing import List
import boto3

# Configuration
BASE_DVC_MODEL_PATH = "model_registry/latest_models"
S3_BUCKET = "bankruptcy-risk-bucket"
S3_PREFIX = "prod-models"

def run_dvc_pull(model_folder: str):
    dvc_file_path = os.path.join(BASE_DVC_MODEL_PATH, f"{model_folder}.dvc")
    logging.info(f"🔄 Pulling {dvc_file_path} from DVC...")
    subprocess.run(["dvc", "pull", dvc_file_path], check=True)

def upload_to_s3(local_path: str, s3_key: str):
    print(f"🚀 Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}")
    s3 = boto3.client("s3")
    s3.upload_file(local_path, S3_BUCKET, s3_key)

def promote_models_to_s3(local_classifier_path:str,local_regressor_paths:List[str]):
    #Pushing the classifier to s3
    classifier_folder=local_classifier_path.split('/')[2]
    run_dvc_pull(classifier_folder)
    s3_classifier_key=f"{S3_PREFIX}/{classifier_folder}/model.pkl"
    upload_to_s3(local_classifier_path,s3_classifier_key)

    #Pushing all the regressor models to S3
    for local_regressor_path in local_regressor_paths:
        regressor_folder=local_regressor_path.split('/')[2]
        run_dvc_pull(regressor_folder)
        s3_regressor_key = f"{S3_PREFIX}/{regressor_folder}/model.pkl"
        upload_to_s3(local_regressor_path, s3_regressor_key)
    logging.info("✅ All models promoted to S3 and ready for Lambda inference.")