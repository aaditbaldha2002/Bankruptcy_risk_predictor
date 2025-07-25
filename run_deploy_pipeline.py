# run_pipeline.py

from pipelines.deployment_pipeline import deployment_pipeline
import sys
sys.path.insert(0, "src")

if __name__ == "__main__":
    deployment_pipeline(
        train_data_path='data/raw/train_data.csv',
        test_data_path='data/raw/test_data.csv'
    )
