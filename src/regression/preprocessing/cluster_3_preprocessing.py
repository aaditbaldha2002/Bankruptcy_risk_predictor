import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(3)
def cluster_3_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)

    sc=StandardScaler()
    indexes=dataset['Index']
    bankrupt_=dataset['Bankrupt?']
    dataset=dataset.drop(columns=['Quick Assets/Current Liability',
                                  "PCA_Net Income to Stockholder's Equity_PCA_Borrowing dependency_Current Liabilities/Equity",
                                  'Fixed Assets to Assets',
                                  'Working capitcal Turnover Rate',
                                  'Quick Ratio',
                                  'Long-term Liability to Current Assets',
                                  'Cash/Current Liability',
                                  ])
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])

    final_df, pca_features, dropped_cols, pca_pairs_df, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.8,
        thresh_high=0.95,
        verbose=True
    )
    final_df['Bankrupt?']=bankrupt_
    
    ARTIFACTS_STORE_DIR='artifacts'
    ARTIFACTS_STORE_DIR=os.path.join('pca')

    joblib.dump(dropped_cols, f'{ARTIFACTS_STORE_DIR}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{ARTIFACTS_STORE_DIR}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{ARTIFACTS_STORE_DIR}/fitted_pca_models.pkl')


    DATA_PATH='artifacts'
    DATA_PATH=os.path.join('cluster_3')
    DATA_PATH=os.path.join('preprocessing')

    final_df.to_csv(DATA_PATH,index=False)
    return DATA_PATH