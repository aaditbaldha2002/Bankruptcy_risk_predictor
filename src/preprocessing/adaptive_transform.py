import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from scipy.stats import skew
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)

def adaptive_transform(data_path: str) -> str:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {data_path} - {fnf_error}")
        raise
    except Exception as e:
        logger.exception(f"Failed to read CSV file at {data_path}: {e}")
        raise

    transformed_data = {}
    transformers = {}

    for feature in tqdm(df.columns, desc="Adaptive Transforming"):
        try:
            x = df[feature].values.reshape(-1, 1)
            feature_skew = skew(x.flatten(), nan_policy='omit')
            min_val = np.nanmin(x)

            transformer = FunctionTransformer(func=None, validate=True)
            transformed = x

            if abs(feature_skew) <= 0.5:
                transformers[feature] = transformer
                transformed_data[feature] = transformed.flatten()
                continue

            if min_val > 0:
                transformer = PowerTransformer(method='box-cox', standardize=True)
                transformed = transformer.fit_transform(x)
            else:
                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformed = transformer.fit_transform(x)

            transformers[feature] = transformer
            transformed_data[feature] = transformed.flatten()

        except Exception as e:
            logger.warning(f"Failed to transform feature '{feature}': {e}")
            transformed_data[feature] = df[feature].values.flatten()
            transformers[feature] = FunctionTransformer(func=None)

    output_dir = 'output/intermediate'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'train_ada_transformed.csv')

    try:
        transformed_df = pd.DataFrame(transformed_data)
        transformed_df.to_csv(file_path, index=False)
        logger.info(f"Transformed data saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save transformed data to {file_path}: {e}")
        raise

    return file_path
