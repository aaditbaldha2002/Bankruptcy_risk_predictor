# run_pipeline.py

from pipelines.deployment_pipeline import deployment_pipeline
import sys
import pandas as pd

from pipelines.inference_pipeline import inference_pipeline
sys.path.insert(0, "src")

if __name__ == "__main__":
    inference_pipeline()
