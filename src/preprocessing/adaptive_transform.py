import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from scipy.stats import skew
from tqdm import tqdm
import os


def adaptive_transform(data_path:str)->str:
    df=pd.read_csv(data_path)
    transformed_data = {}
    transformers = {}

    for feature in tqdm(df.columns, desc="Adaptive Transforming"):
        x = df[feature].values.reshape(-1, 1)
        feature_skew = skew(x.flatten(), nan_policy='omit')
        min_val = np.nanmin(x)

        transformer = FunctionTransformer(func=None, validate=True)
        transformed = x

        if abs(feature_skew) <= 0.5:
            transformers[feature] = transformer
            transformed_data[feature] = transformed.flatten()
            continue

        try:
            if min_val > 0:
                transformer = PowerTransformer(method='box-cox', standardize=True)
                transformed = transformer.fit_transform(x)

            elif min_val <= 0:
                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformed = transformer.fit_transform(x)

            transformers[feature] = transformer
            transformed_data[feature] = transformed.flatten()
        except Exception as e:
            print(f"Warning: Failed to transform {feature} - {e}")
            transformed_data[feature] = x.flatten()
            transformers[feature] = FunctionTransformer(func=None)

    os.makedirs('output/intermediate',exist_ok=True)
    file_path='output/intermediate/train_ada_transformed.csv'
    transformed_data=pd.DataFrame(transformed_data)
    transformed_data.to_csv(file_path,index=False)
    return file_path



