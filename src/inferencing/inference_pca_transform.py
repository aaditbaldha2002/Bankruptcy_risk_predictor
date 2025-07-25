import logging

import numpy as np
import pandas as pd

def inference_pca_transform(pca_pairs_df:pd.DataFrame,df:pd.DataFrame,pca_models)->pd.DataFrame:
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
