import logging
import os

import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(2)
def cluster_2_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)

    sc=StandardScaler()
    indexes=dataset['Index']
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    dataset['Bankrupt?']=bankrupt_



    final_df, pca_features, dropped_cols, pca_pairs_df, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.8,
        thresh_high=0.95,
        verbose=True
    )

    final_df=final_df.drop(columns=['Total debt/Total net worth',
                                    'Cash/Current Liability',
                                    'Long-term Liability to Current Assets',
                                    'Quick Ratio',
                                    'Current Ratio',
                                    'Quick Assets/Current Liability'
                                    ])
    final_df['Bankrupt?']=bankrupt_

    DATA_PATH='artifacts'
    DATA_PATH=os.path.join('cluster_2')
    DATA_PATH=os.path.join('preprocessing')

    final_df.to_csv(DATA_PATH,index=False)
    return DATA_PATH