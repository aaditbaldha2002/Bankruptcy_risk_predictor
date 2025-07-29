import logging
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

def inference_preprocess_data() -> str:
    data_dir=os.path.join('data','raw')
    df=pd.read_csv(os.path.join(data_dir,'inference_test_data.csv'))
    df.drop(columns=['Index','Bankrupt?'],inplace=True)
    df.columns=df.columns.str.strip()
    output_dir = os.path.join('artifacts', 'inferencing')
    os.makedirs(output_dir, exist_ok=True)

    ARTIFACTS_DIR = 'artifacts'
    scaler_path = os.path.join(ARTIFACTS_DIR, 'preprocessing', 'first_scaler.pkl')
    scaler = joblib.load(scaler_path)

    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Drop columns before PCA
    columns_to_drop = joblib.load(os.path.join(ARTIFACTS_DIR, 'preprocessing', 'columns_to_drop_before_pca.pkl'))
    df.drop(columns=columns_to_drop, inplace=True)

    # Load PCA components
    PCA_DIR = os.path.join(ARTIFACTS_DIR, 'preprocessing', 'pca')
    dropped_cols = joblib.load(os.path.join(PCA_DIR, 'columns_to_drop.pkl'))
    pca_pairs_df = joblib.load(os.path.join(PCA_DIR, 'pca_pairs_used.pkl'))
    pca_models = joblib.load(os.path.join(PCA_DIR, 'fitted_pca_models.pkl'))

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
    columns_to_drop_after_pca = joblib.load(os.path.join(PCA_DIR, 'columns_to_drop_after_pca.pkl'))
    df.drop(columns=columns_to_drop_after_pca, inplace=True)

    # Load adaptive transformers
    ADAPTIVE_TRANSFORM_DIR = os.path.join(ARTIFACTS_DIR, 'preprocessing', 'transformed')
    transformers_path = os.path.join(ADAPTIVE_TRANSFORM_DIR, 'adaptive_transformers.pkl')
    try:
        transformers = joblib.load(transformers_path)
        logging.info(f"Loaded adaptive transformers from {transformers_path}")
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

    transformed_data_csv_path = os.path.join(output_dir, 'transformed_input_data.csv')
    transformed_df.to_csv(transformed_data_csv_path, index=False)

    return transformed_data_csv_path
