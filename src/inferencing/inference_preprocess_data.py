import logging
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

def inference_preprocess_data(data:List[float])->str:

    output_dir=os.path.join('../../artifacts','inferencing')
    os.makedirs(output_dir,exist_ok=True)

    df_train = pd.read_csv('../data/raw/train_data.csv')
    df_train_columns=df_train.columns.to_list()
    df = pd.DataFrame([data], columns=[f'{col_name}' for col_name in df_train_columns])
    df.columns=df.columns.str.strip()

    ARTIFACTS_DIR='../../artifacts/'
    scaler_dir=os.path.join(ARTIFACTS_DIR,'preprocessing','first_scaler.pkl')
    scaler = joblib.load(scaler_dir)
    scaler.transform(df)

    columns_to_drop=joblib.load(os.path.join(ARTIFACTS_DIR,'preprocessing','columns_to_drop_before_pca.pkl'))
    df=df.drop(columns=columns_to_drop)
    
    PCA_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing','pca')

    dropped_cols=joblib.load(PCA_DIR,'columns_to_drop.pkl')
    pca_pairs_df=joblib.load(PCA_DIR,'pca_pairs_used.pkl')
    pca_models=joblib.load(PCA_DIR,'fitted_pca_models.pkl')

    df=df.drop(columns=[col for col in dropped_cols if col in df.columns], inplace=True)
    new_cols = []

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
        
    columns_to_drop_after_pca=joblib.load(os.path.join(PCA_DIR,'columns_to_drop_after_pca.pkl'))
    df=df.drop(columns=columns_to_drop_after_pca)
    ADAPTIVE_TRANSFORM_DIR=os.path.join(ARTIFACTS_DIR,'preprocessing','transformed')

    try:
        transformers = joblib.load(os.path.join(ADAPTIVE_TRANSFORM_DIR,'adaptive_transformers.pkl'))
        logging.info(f"Loaded adaptive transformers from {os.path.join(ADAPTIVE_TRANSFORM_DIR,'adaptive_transformers.pkl')}")
    except Exception as e:
        logging.exception(f"Could not load transformers from {os.path.join(ADAPTIVE_TRANSFORM_DIR,'adaptive_transformers.pkl')}: {e}")
        raise

    # Check for missing features
    missing_features = [f for f in transformers.keys() if f not in df.columns]
    if missing_features:
        raise ValueError(f"Inference data is missing the following required features: {missing_features}")

    transformed_data = {}

    for feature, transformer in transformers.items():
        try:
            x = df[feature].values.reshape(-1, 1)
            transformed_feature = transformer.transform(x)
            transformed_data[feature] = transformed_feature.flatten()
        except Exception as e:
            logging.exception(f"Failed to transform feature '{feature}': {e}")
            raise RuntimeError(f"Transformation failed for feature '{feature}': {e}")
    
    transformed_data_file_path=os.path.join(output_dir,'transformed_input_data.pkl')
    joblib.dump(transformed_data,transformed_data_file_path)

    return transformed_data_file_path