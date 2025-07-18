import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(4)
def cluster_4_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    sc=StandardScaler()
    indexes=dataset['Index']
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])

    final_df, pca_features, dropped_cols, pca_pairs_df, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.9,
        thresh_high=0.95,
        verbose=True
    )

    DATA_PATH='artifacts'
    DATA_PATH=os.path.join('cluster_4')
    DATA_PATH=os.path.join('preprocessing')

    joblib.dump(dropped_cols, f'{DATA_PATH}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{DATA_PATH}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{DATA_PATH}/fitted_pca_models.pkl')


    final_df['Bankrupt?']=bankrupt_


    final_df.to_csv(DATA_PATH)
    return DATA_PATH