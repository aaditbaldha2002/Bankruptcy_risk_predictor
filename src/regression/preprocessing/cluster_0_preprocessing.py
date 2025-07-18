import logging
import os
import joblib
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor

import pandas as pd
from sklearn.preprocessing import StandardScaler

@register_preprocessor(0)
def cluster_0_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)

    sc=StandardScaler()
    indexes=dataset['Index']
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    dataset=dataset.drop(columns=['Index','Bankrupt?','Cash/Total Assets','Total debt/Total net worth','Equity to Long-term Liability','Cash/Current Liability','Long-term Liability to Current Assets','Quick Ratio','Working capitcal Turnover Rate','Current Ratio','Quick Assets/Current Liability'])

    final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.9,
        thresh_high=0.95,
        verbose=True
    )

    if all_pca_pairs:
        pca_pairs_df = pd.concat(all_pca_pairs, ignore_index=True)
    else:
        pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

    ARTIFACTS_STORE_DIR='artifacts'
    ARTIFACTS_STORE_DIR=os.path.join('pca')

    joblib.dump(dropped_cols, f'{ARTIFACTS_STORE_DIR}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{ARTIFACTS_STORE_DIR}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{ARTIFACTS_STORE_DIR}/fitted_pca_models.pkl')

    final_df['Bankrupt?']=bankrupt_
    DATA_PATH='artifacts'
    DATA_PATH=os.path.join('cluster_0')
    DATA_PATH=os.path.join('preprocessing')

    final_df.to_csv(DATA_PATH,index=False)
    return DATA_PATH